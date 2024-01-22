import torch
from torch.utils.data import Dataset
import torchaudio
import random
import numpy as np
import csv
import os
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
)
import torchvision.transforms as T
from torchvision.transforms._transforms_video import NormalizeVideo
from transforms import SpatialCrop, TemporalCrop
from randomerasing import RandomErasing
from rand_auto_aug import RandAugment
from mmaction.datasets.pipelines import Compose
from PIL import Image

def get_video_clip(num_frames, clip_len = 32, train_mode = False):
    """
    clip_len: int, the number of frames in the output video clip
    num_frames: int, the total number of frames for the input video
    test_mode: bool
    """

    seg_size = float(num_frames - 1) / clip_len
    seq = []
    for i in range(clip_len):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i+1)))

        if train_mode:
            seq.append(random.randint(start, end))
        else:
            seq.append((start + end) // 2)
    
    return np.array(seq)

def get_verb_classes(annotation_path="epic-annotations/EPIC_100_verb_classes.csv"):
    verb_dict = {}
    with open(annotation_path) as f:
        f_csv = csv.reader(f)
        for i, row in enumerate(f_csv):
            if i == 0:
                continue
            verb_dict[row[1]] = int(row[0])
    f.close()
    return verb_dict


def get_noun_classes(annotation_path="epic-annotations/EPIC_100_noun_classes.csv"):
    noun_dict = {}
    with open(annotation_path) as f:
        f_csv = csv.reader(f)
        for i, row in enumerate(f_csv):
            if i == 0:
                continue
            noun_dict[row[1]] = int(row[0])
    f.close()
    return noun_dict


class EPICKitchensTrain(Dataset):
    def __init__(
        self,
        audio_conf,
        split="train",
        audio_data_path="/path/to/EPIC-KITCHENS-Audio/",
        rgb_data_path = "/path/to/EPIC-KITCHENS/",
        num_position=512,
    ):

        self.audio_conf = audio_conf
        self.audio_data_path = audio_data_path
        self.rgb_data_path = rgb_data_path
        self.num_position = num_position

        self.get_audio_parameters()

        self.num_frames = 32
        self.video_transform = ApplyTransformToKey(
            key="video",
            transform=T.Compose(
                [
                    T.Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=256),
                    NormalizeVideo(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    TemporalCrop(frames_per_clip=self.num_frames, stride=40),
                    SpatialCrop(crop_size=224, num_crops=1),
                ]
            ),
        )
        self.random_erasing = RandomErasing(device="cpu", probability=0.25)
        self.random_augment = RandAugment(
            magnitude=9, magnitude_std=0.5, increasing_severity=True
        )

        verb_dict = get_verb_classes()
        noun_dict = get_noun_classes()

        self.action_map = {}
        with open("epic-annotations/epic_action_classes.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                self.action_map[
                    str(verb_dict[row[0]]) + "+" + str(noun_dict[row[1]])
                ] = i
        f.close()

        self.available_list = []
        with open("epic-annotations/available_sound.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                self.available_list.append(row[0])
        f.close()

        csv_file_path1 = "epic-annotations/EPIC_100_train_full_half1.csv"
        csv_file_path2 = "epic-annotations/EPIC_100_train_full_half2.csv"

        self.sample_dict = {}
        with open(csv_file_path1) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                self.sample_dict[row[0]] = ["RGB"]
        f.close()

        with open(csv_file_path2) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                self.sample_dict[row[0]] = ["Audio"]
        f.close()

        samples = []
        with open("epic-annotations/EPIC_100_train.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i == 0:
                    continue
                if row[2] not in self.available_list and row[0] in self.sample_dict and "Audio" in self.sample_dict[row[0]]:
                    continue
                if row[0] not in self.sample_dict:
                    continue
                samples.append(row)
        f.close()
        with open("epic-annotations/EPIC_100_validation.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i == 0:
                    continue
                if row[2] not in self.available_list and row[0] in self.sample_dict and "Audio" in self.sample_dict[row[0]]:
                    continue
                if row[0] not in self.sample_dict:
                    continue
                samples.append(row)
        f.close()

        self.data = samples
        self.noun_label_num = 300
        self.verb_label_num = 97

    def get_audio_parameters(self):
        self.audio_melbins = self.audio_conf.get("num_mel_bins")
        self.audio_freqm = self.audio_conf.get("freqm")
        self.audio_timem = self.audio_conf.get("timem")
        self.audio_mixup = self.audio_conf.get("mixup")
        self.audio_dataset = self.audio_conf.get("dataset")
        # dataset spectrogram mean and std, used to normalize the input
        self.audio_norm_mean = self.audio_conf.get("mean")
        self.audio_norm_std = self.audio_conf.get("std")
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.audio_skip_norm = (
            self.audio_conf.get("skip_norm")
            if self.audio_conf.get("skip_norm")
            else False
        )
        # if add noise for data augmentation
        self.audio_noise = self.audio_conf.get("noise")

    def _wav2fbank(
        self,
        filename,
    ):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.audio_melbins,
            dither=0.0,
            frame_shift=10,
        )
        target_length = self.audio_conf.get("target_length")
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            start_idx = np.random.randint(0, -p, (1,))[0]
            fbank = fbank[start_idx : start_idx + target_length, :]

        return fbank


    def get_fbank(self, datum):
        audio_path = os.path.join(self.audio_data_path, datum[1], datum[0] + ".wav")
        fbank = self._wav2fbank(audio_path)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.audio_freqm)
        timem = torchaudio.transforms.TimeMasking(self.audio_timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.audio_freqm != 0:
            fbank = freqm(fbank)
        if self.audio_timem != 0:
            fbank = timem(fbank.unsqueeze(0))
            fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.audio_skip_norm:
            fbank = (fbank - self.audio_norm_mean) / (self.audio_norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.audio_noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        return fbank

    def get_rgb_frames(self, datum):
        start_frame = int(datum[6])
        end_frame = int(datum[7])
        video_path = os.path.join(
            self.rgb_data_path, datum[1], "rgb_frames", datum[2]
        )
        indices = get_video_clip(end_frame-start_frame, clip_len = self.num_frames, train_mode = True ) + start_frame
        indices = torch.clamp(indices, start_frame, end_frame).numpy()

        frames = []
        for indice in list(indices):
            img = Image.open(os.path.join(video_path, "frame_%010d.jpg" % indice))
            # img = self.random_augment(img)
            frames.append(np.array(img))
        frames = np.stack(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = torch.Tensor(frames)

        # -------- RandomErasing ----------
        # frames = self.random_erasing(frames)

        frames = frames.transpose(0, 1)
        # -------- Transform ---------
        video_data = {}
        video_data["video"] = frames
        video_data = self.video_transform(video_data)
        video_data = torch.stack(video_data["video"]).squeeze(0)

        return video_data

    def get_label(self, datum):
        action_key = datum[-5] + "+" + datum[-3]
        action_label = self.action_map[action_key]

        return action_label

    def __getitem__(self, index):
        datum = self.data[index]
        action_label = self.get_label(datum)
        available_modalities = self.sample_dict[datum[0]]
        output_data = {"Audio": None, "RGB": None}
        if "Audio" in available_modalities:
            output_data["Audio"] = self.get_fbank(datum)
            audio_mask = np.ones((self.num_position, 1))
            audio_pseudo = torch.Tensor(np.load("audio_pseudo/" + datum[0] + ".npy"))
            audio_pseudo = torch.mean(audio_pseudo, dim=0)
        else:
            output_data["Audio"] = torch.rand(128, 128)
            audio_mask = np.zeros((self.num_position, 1))
            audio_pseudo = torch.zeros((3806,))

        # -------------------------------------------------- RGB -------------------------------------------
        if "RGB" in available_modalities:
            output_data["RGB"] = self.get_rgb_frames(datum)
            rgb_mask = np.ones((self.num_position, 1))
            rgb_pseudo = torch.Tensor(np.load("rgb_pseudo/" + datum[0] + ".npy"))
            rgb_pseudo = torch.mean(rgb_pseudo, dim=0)
        else:
            output_data["RGB"] = torch.rand(3, self.num_frames, 224, 224)
            rgb_mask = np.zeros((self.num_position, 1))
            rgb_pseudo = torch.zeros((3806, ))
        
        masks = {
            "RGB": rgb_mask.astype(np.float32),
            "Audio": audio_mask.astype(np.float32),
        }

        return (
            output_data,
            action_label,
            masks,
            audio_pseudo,
            rgb_pseudo
        )

    def __len__(self):
        return len(self.data)
