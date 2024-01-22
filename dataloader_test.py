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

class EPICKitchensTest(Dataset):
    def __init__(self, audio_conf, split='test', audio_data_path = '/path/to/EPIC-KITCHENS-Audio/', cfg=None, num_position = 512):
        self.audio_conf = audio_conf
        self.audio_data_path = audio_data_path
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

        test_pipeline = cfg.data.test.pipeline
        self.test_pipeline = Compose(test_pipeline)
        self.cfg = cfg

        verb_dict = get_verb_classes()
        noun_dict = get_noun_classes()

        self.action_map = {}
        with open("epic-annotations/epic_action_classes.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                self.action_map[str(verb_dict[row[0]]) + "+" + str(noun_dict[row[1]])] = i
        f.close()

        self.available_list = []
        with open("epic-annotations/available_sound.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                self.available_list.append(row[0])
        f.close()

        self.sample_dict = {}
        with open("epic-annotations/EPIC_100_full_test.csv") as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                self.sample_dict[row[0]] = ["Audio", "RGB"]
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
        self.audio_melbins = self.audio_conf.get('num_mel_bins')
        self.audio_freqm = self.audio_conf.get('freqm')
        self.audio_timem = self.audio_conf.get('timem')
        self.audio_mixup = self.audio_conf.get('mixup')
        self.audio_dataset = self.audio_conf.get('dataset')
        # dataset spectrogram mean and std, used to normalize the input
        self.audio_norm_mean = self.audio_conf.get('mean')
        self.audio_norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.audio_skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        # if add noise for data augmentation
        self.audio_noise = self.audio_conf.get('noise')

    def _wav2fbank(self, filename, start_time, end_time,):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.audio_melbins, dither=0.0, frame_shift=10)
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames
        stride = int(n_frames / 3.1)

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbanks = m(fbank)
            fbanks = fbanks.unsqueeze(0)
        elif p < 0:
            # fbanks = fbank[:target_length].unsqueeze(0)
            ids = np.arange(start=0, stop=-p, step=stride)
            fbanks = []
            for id in list(ids):
                fbank1 = fbank[id : id + target_length, :]
                fbanks.append(fbank1)
            fbanks = torch.stack(fbanks)
        else:
            fbanks = fbank.unsqueeze(0)
        
        return fbanks

    def get_fbank(self, index):
        datum = self.data[index]
        audio_path = os.path.join(self.audio_data_path, datum[1], datum[0] + ".wav")
        fbank = self._wav2fbank(audio_path)

        # normalize the input for both training and test
        if not self.audio_skip_norm:
            fbank = (fbank - self.audio_norm_mean) / (self.audio_norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        return fbank
    
    def get_rgb_frames(self, index):
        datum = self.data[index]
        start_frame = int(datum[6])
        end_frame = int(datum[7])
        video_path = os.path.join('/path/to/EPIC-KITCHENS/', datum[1], 'rgb_frames', datum[2])

        indices = torch.linspace(
            start_frame, end_frame, self.num_frames
        )
        indices = torch.clamp(
            indices, start_frame, end_frame
        ).numpy()
        frames = []
        for indice in list(indices):
            img = Image.open(os.path.join(video_path, "frame_%010d.jpg" % indice))
            frames.append(np.array(img))
        frames = np.stack(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = torch.Tensor(frames)

        frames = frames.transpose(0, 1)
        # -------- Transform ---------
        video_data = {}
        video_data["video"] = frames
        video_data = self.video_transform(video_data)
        video_data = torch.stack(video_data["video"]).squeeze(
            0
        )

        return video_data

    def get_label(self, index):
        datum = self.data[index]
        action_key = datum[-5] + "+" + datum[-3]
        action_label = self.action_map[action_key]
        
        return action_label

    def __getitem__(self, index):
        available_modalities = ["RGB", "Audio"]
        output_data = {"Audio": None, "RGB": None}
        if "Audio" in available_modalities:
            output_data["Audio"] = self.get_fbank(index)
            audio_mask = np.ones((self.num_position, 1))
        else:
            output_data["Audio"] = torch.rand(128, 128)
            audio_mask = np.zeros((self.num_position, 1))

        # -------------------------------------------------- RGB -------------------------------------------
        if "RGB" in available_modalities:
            output_data['RGB'] = self.get_rgb_frames(index).unsqueeze(0)
            rgb_mask = np.ones((self.num_position, 1))
        else:
            output_data['RGB'] = torch.rand(1, self.num_frames, 224, 224)
            rgb_mask = np.zeros((self.num_position, 1))

        action_label = self.get_label(index)
        num_of_fbanks = output_data["Audio"].size()[0]
        new_output_data = {'Audio': [], 'RGB': [],}
        for i in range(1):
            rgb_clip = output_data['RGB'][i:i+1]
            rgb_clip = torch.tile(rgb_clip, (num_of_fbanks, 1, 1, 1, 1))
            new_output_data['RGB'].append(rgb_clip)
            new_output_data['Audio'].append(output_data["Audio"])
        new_output_data['RGB'] = torch.cat(new_output_data['RGB'], dim=0)
        new_output_data['Audio'] = torch.cat(new_output_data['Audio'], dim=0)
        
        rgb_mask = np.ones((new_output_data['RGB'].size()[0], self.num_position, 1))
        audio_mask = np.ones((new_output_data['RGB'].size()[0], self.num_position, 1))
        flow_mask = np.zeros((new_output_data['RGB'].size()[0], self.num_position, 1))

        return new_output_data["RGB"], new_output_data["Audio"], action_label, rgb_mask.astype(np.float32), audio_mask.astype(np.float32)

    def __len__(self):
        return len(self.data)
