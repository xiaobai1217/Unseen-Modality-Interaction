import torch
from dataloader_test import EPICKitchensTest
from ast_model import ASTModel
import pdb
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
from omnivore_models.omnivore_model import omnivore_swinB_imagenet21k, omnivore_swinT
from vit import ViT
from feature_reorganization import ViTReorganization
from train import spatialtemporal2tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_position",
        type=int,
        help="path to the checkpoint",
        default=512,
    )
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    rgb_model.load_state_dict(checkpoint['state_dict'])
    rgb_model = rgb_model.to(device)
    rgb_model.eval()

    multimodal_model = ViT(num_classes = 3806, dim = 256, depth = 6, heads = 8, mlp_dim = 512, num_position = args.num_position)
    multimodal_model = torch.nn.DataParallel(multimodal_model)
    multimodal_model = multimodal_model.to(device)

    reorganization_module = ViTReorganization(dim = 256, depth = 6, heads = 8, mlp_dim = 512, num_position = args.num_position)
    reorganization_module = torch.nn.DataParallel(reorganization_module)
    reorganization_module = reorganization_module.to(device)

    checkpoint = torch.load("checkpoints/best_multimodal1e-3resumed.pt")
    multimodal_model.load_state_dict(checkpoint['model'])
    multimodal_model.eval()

    reorganization_module.load_state_dict(checkpoint['reorganization'])
    reorganization_module.eval()

    test_dataset = EPICKitchensTest(audio_conf=val_audio_configs, split="test", num_position = args.num_position)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    num_of_samples = len(test_dataloader)
    acc = 0
    save_path = 'predictions/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    num1 = 5
    for i, (rgb_data, audio, action_label, rgb_mask, audio_mask) in enumerate(test_dataloader):
        audio = audio.cuda().squeeze(0)
        rgb_data = rgb_data.cuda().squeeze(0)
        rgb_mask = rgb_mask.cuda().squeeze(0)
        audio_mask = audio_mask.cuda().squeeze(0)

        output_predictions = []
        with torch.no_grad():
            for k in range(audio.size()[0] // num1 + 1):
                if k*num1 >= audio.size()[0]:
                    break
                audio_outputs = audio_model(audio[k*num1:(k+1)*num1]) # [batch_size, 146, 768]
                rgb_outputs = rgb_model(rgb_data[k*num1:(k+1)*num1]) #[batch_size, 768, 16, 7, 7]

                rgb_outputs = spatialtemporal2tokens(rgb_outputs)

                rgb_outputs, audio_outputs = reorganization_module(rgb_outputs, audio_outputs)
                audio_mask[:,:,:] = 0.0
                outputs = multimodal_model(rgb_outputs, audio_outputs, rgb_mask, audio_mask)
                
                outputs = torch.softmax(outputs, dim=-1)
                output_predictions.append(outputs.detach())
        predictions = torch.cat(output_predictions, dim=0)
        predictions = torch.mean(predictions, dim=0)
        predictions = predictions.detach().cpu().numpy()
        action_label = action_label.numpy()[0]

        if np.argmax(predictions) == action_label:
            acc += 1
        print(i+1, '/', num_of_samples, 'Accuracy:', acc / (i+1))
        