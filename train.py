from ast_model import ASTModel
import pdb
from dataloader_train import EPICKitchensTrain
from dataloader_validation import EPICKitchensValidation
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
import warnings
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import datetime
from ast_configs import get_audio_configs
from mmaction.apis import init_recognizer, inference_recognizer
from omnivore_models.omnivore_model import omnivore_swinB_imagenet21k, omnivore_swinT
from vit import ViT
from feature_reorganization import ViTReorganization

def dict_to_cuda(data):
    for key, value in data.items():
        data[key] = value.cuda()
    return data

def spatialtemporal2tokens(data):
    b, c, f, h, w = data.size()
    data = data.view(b, c, f * h * w)
    data = data.transpose(1, 2).contiguous()
    return data

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.0)
        weight.scatter_(-1, target.unsqueeze(-1), (1.0 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class AlignmentModule(nn.Module):
    def __init__(self, dim=256):
        super(AlignmentModule, self).__init__()
        self.base_vectors = nn.Parameter(torch.randn(1, 3806, 256))
    
    def forward(self, input):
        # input [batchsize, n, 256]
        input = torch.mean(input, dim=1, keepdim=True) # [batch_size, 1, 256]
        base_vectors = self.base_vectors.repeat(input.size()[0], 1, 1)
        sim = torch.mean((base_vectors - input) ** 2, dim=-1)
        return sim

def extract_features(unimodal_models, data):
    outputs = {}
    for key, value in data.items():
        outputs[key] = unimodal_models[key](value)
        if key == 'RGB':
            outputs[key] = spatialtemporal2tokens(outputs[key])
    return outputs

def train_one_step(
    data,
    labels,
    masks,
    audio_pseudo,
    rgb_pseudo,
    unimodal_models,
    multimodal_model,
    reorganization_module,
    alignment_model,
    optim,
    loss_fn,
    kl_loss_fn,
    scaler,
):
    with torch.no_grad():
        outputs = extract_features(unimodal_models, data)

    rgb, audio = reorganization_module(
        outputs['RGB'], outputs['Audio']
    )

    audio_sim = alignment_model(audio)
    rgb_sim = alignment_model(rgb)

    outputs = multimodal_model(
        rgb, audio, masks['RGB'], masks['Audio']
    )

    audio_indices = torch.sum(masks['Audio'].squeeze(-1), dim=-1) > 0
    rgb_indices = torch.sum(masks['RGB'].squeeze(-1), dim=-1) > 0

    audio_labels = labels[audio_indices]
    rgb_labels = labels[rgb_indices]
    audio_onehot_labels = F.one_hot(audio_labels, num_classes = 3806)
    rgb_onehot_labels = F.one_hot(rgb_labels, num_classes = 3806)

    audio_sim = audio_sim[audio_indices]
    rgb_sim = rgb_sim[rgb_indices]

    audio_sim = torch.sum(audio_sim * audio_onehot_labels, dim=-1)
    rgb_sim = torch.sum(rgb_sim*rgb_onehot_labels, dim=-1)
    alignment_loss = (torch.sum(audio_sim) + torch.sum(rgb_sim)) / (torch.sum(audio_indices) + torch.sum(rgb_indices))

    audio_pseudo = audio_pseudo[audio_indices]
    rgb_pseudo = rgb_pseudo[rgb_indices]
    probs = torch.softmax(outputs[:,1], dim=-1)
    audio_prob = probs[audio_indices]
    rgb_prob = probs[rgb_indices]

    kl_loss = torch.mean(kl_loss_fn(torch.log(audio_prob), audio_pseudo)) * torch.sum(audio_indices) + torch.mean(kl_loss_fn(torch.log(rgb_prob), rgb_pseudo)) * torch.sum(rgb_indices)
    loss = loss_fn(outputs[:,0], labels) + kl_loss / labels.size()[0] * 3000 +  0.001 * alignment_loss

    optim.zero_grad()
    optim.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    return outputs, loss


def val_one_step(
    data,
    labels,
    masks,
    unimodal_models,
    multimodal_model,
    reorganization_module,
    alignment_model,
    loss_fn,
):
    with torch.no_grad():
        outputs = extract_features(unimodal_models, data)

        rgb, audio = reorganization_module(
            outputs['RGB'], outputs['Audio']
        )
        outputs = multimodal_model(
            rgb, audio, masks['RGB'], masks['Audio']
        )
        loss = (loss_fn(outputs[:,0], labels) + loss_fn(outputs[:,1], labels)) * 0.5

    return outputs, loss


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, help="learning rate", default=1e-4
    )  # 3e-5 first best,8e-6 2nd, 2e-6 3rd
    parser.add_argument("--batch_size", type=int, help="batch size", default=12)
    parser.add_argument(
        "--audio_data_path",
        type=str,
        help="path to data",
        default="/path/to/EPIC-KITCHENS-Audio-Clips/",
    )
    parser.add_argument(
        "--rgb_data_path",
        type=str,
        help="path to data",
        default="/path/to/EPIC-KITCHENS/",
    )
    parser.add_argument(
        "--save_name", type=str, help="name to save the model", default="default",
    )
    parser.add_argument(
        "--resume_training", type=bool, help="resume training or not", default=False
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        help="path to the checkpoint",
        default="checkpoints/best.pt",
    )
    parser.add_argument(
        "--num_position", type=int, help="path to the checkpoint", default=512,
    )
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    warnings.filterwarnings("ignore")

    # assign the desired device.
    device = "cuda"  # or 'cpu'
    device = torch.device(device)

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    batch_size = args.batch_size
    target_length = 128
    train_audio_configs, val_audio_configs = get_audio_configs(
        target_length=target_length
    )

    audio_model = ASTModel(
        label_dim=3806,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=target_length,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
    )
    audio_model = audio_model.to(device)
    audio_model = nn.DataParallel(audio_model)
    checkpoint = torch.load("checkpoints/audio.pt")
    audio_model.load_state_dict(checkpoint["model"])
    audio_model.eval()

    rgb_model = omnivore_swinT(pretrained=False)
    rgb_model.heads = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=3806, bias=True)
    )
    rgb_model.multimodal_model = False
    rgb_model = torch.nn.DataParallel(rgb_model)
    checkpoint = torch.load("checkpoints/rgb.pt")
    rgb_model.load_state_dict(checkpoint["state_dict"])
    rgb_model = rgb_model.to(device)
    rgb_model.eval()

    unimodal_models = {
        'RGB': rgb_model,
        'Audio': audio_model
    }

    multimodal_model = ViT(
        num_classes=3806,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        num_position=args.num_position,
    )
    multimodal_model = torch.nn.DataParallel(multimodal_model)
    multimodal_model = multimodal_model.to(device)

    reorganization_module = ViTReorganization(
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        num_position=args.num_position,
    )
    reorganization_module = torch.nn.DataParallel(reorganization_module)
    reorganization_module = reorganization_module.to(device)

    alignment_model = AlignmentModule()
    alignment_model = torch.nn.DataParallel(alignment_model)
    alignment_model = alignment_model.to(device)

    if args.resume_training is True:
        checkpoint = torch.load(args.resume_checkpoint)
        multimodal_model.load_state_dict(checkpoint["model"])
        reorganization_module.load_state_dict(checkpoint["reorganization"])
        alignment_model.load_state_dict(checkpoint["alignment"])

    loss_fn = LabelSmoothLoss(smoothing=0.1)
    loss_fn = loss_fn.cuda()

    kl_loss_fn = nn.KLDivLoss(reduce=False)
    kl_loss_fn = kl_loss_fn.cuda()

    optim = torch.optim.SGD(
        list(multimodal_model.parameters())+list(position_model.parameters()) + list(alignment_model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    if args.resume_training is True:
        optim.load_state_dict(checkpoint['optimizer'])

    train_loader = torch.utils.data.DataLoader(
        EPICKitchensTrain(
            audio_conf=train_audio_configs,
            audio_data_path = args.audio_data_path,
            rgb_data_path = args.rgb_data_path,
            num_position=args.num_position,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        EPICKitchensValidation(
            audio_conf=val_audio_configs,
            split="validation",
            audio_data_path = args.audio_data_path,
            rgb_data_path = args.rgb_data_path,
            num_position=args.num_position,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    dataloaders = {"train": train_loader, "val": val_loader}
    BestLoss = float("inf")
    BestEpoch = 0
    BestAcc = 0
    scaler = GradScaler()

    log_path = "logs/{}.csv".format(args.save_name)
    print("---------------Start Training---------------")
    with open(log_path, "a") as f:
        for epoch_i in range(200):
            print("Epoch: %02d" % epoch_i)
            for split in ["train", "val"]:
                acc = 0
                count = 0
                total_loss = 0
                loss = 0
                print(split)
                multimodal_model.train(split == "train")
                position_model.train(split == "train")
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:

                    for (
                        i,
                        (
                            data,
                            labels,
                            masks,
                            audio_pseudo,
                            rgb_pseudo
                        ),
                    ) in enumerate(dataloaders[split]):
                        data = dict_to_cuda(data)
                        masks = dict_to_cuda(masks)
                        labels = labels.cuda()
                        audio_pseudo = audio_pseudo.cuda()
                        rgb_pseudo = rgb_pseudo.cuda()

                        if split == "train":
                            outputs, loss = train_one_step(
                                data,
                                labels,
                                masks,
                                audio_pseudo,
                                rgb_pseudo,
                                unimodal_models,
                                multimodal_model,
                                reorganization_module,
                                alignment_model,
                                optim,
                                loss_fn,
                                kl_loss_fn,
                                scaler,
                            )
                        else:
                            outputs, loss = val_one_step(
                                data,
                                labels,
                                masks,
                                unimodal_models,
                                multimodal_model,
                                reorganization_module,
                                alignment_model,
                                loss_fn,
                            )

                        total_loss += loss.item() * batch_size

                        outputs = torch.softmax(outputs, dim=-1)
                        outputs = torch.mean(outputs, dim=1)
                        _, predict = torch.max(outputs, dim=1)
                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)

                        count += outputs.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(
                                total_loss / float(count),
                                loss.item(),
                                acc / float(count),
                            )
                        )
                        pbar.update()
                    f.write(
                        "{},{},{},{}\n".format(
                            epoch_i,
                            split,
                            total_loss / float(count),
                            acc / float(count),
                        )
                    )
                    f.flush()

            if acc / float(count) > BestAcc:
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                BestAcc = acc / float(count)
                save = {
                    "epoch": epoch_i,
                    "model": multimodal_model.state_dict(),
                    "reorganization": reorganization_module.state_dict(),
                    "alignment": alignment_model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "best_loss": BestLoss,
                    "best_acc": BestAcc,
                }

                torch.save(
                    save, base_path + "best_multimodal{}.pt".format(args.save_name)
                )
    f.close()
