import gc
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ezsynth.constants import DEFAULT_FLOW_ARCH, DEFAULT_FLOW_MODEL
from ezsynth.utils.flow_utils.OpticalFlow import RAFT_flow



from consistencer import deflicker
from ezsynth.pickleing import load_from_pickle

stylized_frames = load_from_pickle("stylized.pkl")
img_frs_seq = load_from_pickle("img_frs_seq.pkl")

cfg_path = "J:/AI/Ezsynth/consistency/config_flow_100.json"
checkpoint_path = "None"
ckpt_filter = "J:/AI/Ezsynth/consistency/cvpr2023_deflicker_public_folder/pretrained_weights/neural_filter.pth"
ckpt_local = "J:/AI/Ezsynth/consistency/cvpr2023_deflicker_public_folder/pretrained_weights/local_refinement_net.pth"

rafter = RAFT_flow(model_name=DEFAULT_FLOW_MODEL, arch=DEFAULT_FLOW_ARCH)

iters_num: int = 50
down_scale: int = 1
results_folder: str = "output_reconstruct"

deflicker(
    cfg_path,
    checkpoint_path,
    img_frs_seq,
    rafter,
    iters_num,
    down_scale,
    results_folder,
)
