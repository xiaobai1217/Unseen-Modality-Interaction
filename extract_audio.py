import moviepy.editor as mp
import librosa
import soundfile
import glob
import subprocess
import os
import multiprocessing

def obtain_list(source_path):
    files = []
    txt = glob.glob(source_path + '/*.MP4') # '/*.flac'
    for item in txt:
        files.append(item)
    return files

def convert(v, output_path):
    subprocess.check_call([
    'ffmpeg',
    '-n',
    '-i', v,
    '-acodec', 'pcm_s16le',
    '-ac','1',
    '-ar','16000',
    output_path + '%s.wav' % v.split('/')[-1][:-4]])

split_list = ['train', 'test']
domain_list = os.listdir('/path_to/EPIC-KITCHENS/')
domain_list.sort()
save_path = '/path_to/EPIC-KITCHENS-Audio/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

source_path = '/path_to/EPIC-KITCHENS/videos/'

file_list = obtain_list(source_path)
for i, file1 in enumerate(file_list):
    name1 = file1.split('/')[-1][:-4]
    convert(file1, save_path)

