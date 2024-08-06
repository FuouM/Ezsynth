import gc
import os
import sys
import time

import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reconstructor import deflicker_refiner


from ezsynth.pickleing import dump_to_pickle, load_from_pickle

stylized_frames = load_from_pickle("stylized.pkl")
img_frs_seq = load_from_pickle("img_frs_seq.pkl")
video_frames_reconstruction = load_from_pickle("video_frames_reconstruction.pkl")

cfg_path = "J:/AI/Ezsynth/consistency/config_flow_100.json"
checkpoint_path = "None"
ckpt_filter = "J:/AI/Ezsynth/consistency/cvpr2023_deflicker_public_folder/pretrained_weights/neural_filter.pth"
ckpt_local = "J:/AI/Ezsynth/consistency/cvpr2023_deflicker_public_folder/pretrained_weights/local_refinement_net.pth"
output_folder = "output_refined"
os.makedirs(output_folder, exist_ok=True)
# print(video_frames_reconstruction.shape) # (540, 960, 3, 11)
# print(video_frames_reconstruction) # 0.09686524 0.0900023  0.09165257 ... 0.13096201 0.12540334

# print(stylized_frames[0].shape) # (540, 960, 3)
# print(stylized_frames[0]) # [[[ 39  36  31]

deflicker_refiner(
    ckpt_filter,
    ckpt_local,
    stylized_frames,
    video_frames_reconstruction,
    seed=2023,
    output_folder=output_folder,
)
