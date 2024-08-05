import gc
import os
import sys
import time

import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ezsynth.sequences import EasySequence
from ezsynth.aux_classes import RunConfig
from ezsynth.aux_utils import save_seq
from ezsynth.main_ez import Ezsynth

st = time.time()

style_paths = [
    "J:/AI/Ezsynth/examples/styles/style000.jpg",
]

image_folder = "J:/AI/Ezsynth/examples/input"
mask_folder = "J:/AI/Ezsynth/examples/mask/mask_feather"
output_folder = "J:/AI/Ezsynth/output"

edge_method="Classic"

flow_arch = "RAFT"
flow_model = "sintel"

ezrunner = Ezsynth(
    style_paths=style_paths,
    image_folder=image_folder,
    cfg=RunConfig(pre_mask=False, feather=0),
    edge_method=edge_method,
    raft_flow_model_name=flow_model,
    mask_folder=mask_folder,
    # do_mask=True,
    do_mask=False,
    flow_arch=flow_arch
)

only_mode = EasySequence.MODE_FWD

stylized_frames, err_frames, flow_frames = ezrunner.run_sequences_full(
    only_mode, return_flow=False
)

save_seq(stylized_frames, "J:/AI/Ezsynth/output_preconstruct")

from ezsynth.pickleing import dump_to_pickle

dump_to_pickle(stylized_frames, "stylized.pkl")
dump_to_pickle(ezrunner.img_frs_seq, "img_frs_seq.pkl")

gc.collect()
torch.cuda.empty_cache()

print(f"Time taken: {time.time() - st:.4f} s")
