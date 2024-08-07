import torch
import numpy as np
import tqdm
import cv2
import random

from consistency.network_filter import UNet
from consistency.network_local import TransformNet
from consistency.easydict import EasyDict as edict
from ezsynth.utils.flow_utils.core.utils.utils import InputPadder


def deflicker_refiner(
    ckpt_filter: str,
    ckpt_local: str,
    stylized: list[np.ndarray],
    video_frames_reconstruction: np.ndarray,
    seed=2023,
    output_folder="output_refined",
    output_folder_concat="output_refined_concat"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Define neural filter model
    filter_net = UNet(in_channels=6, out_channels=3, init_features=32)

    ### Define local refinement model
    ### Local refinement net is rely on Lai et al., ECCV 2018, thank you!
    model_opts = edict({"nf": 32, "norm": "IN", "model": "TransformNet", "blocks": 5})
    local_net = TransformNet(model_opts, nc_in=12, nc_out=3)

    ### load neural filter model
    ckpt = torch.load(ckpt_filter)
    filter_net.load_state_dict(ckpt)
    filter_net.to(device)
    filter_net.eval()

    ### load  local refinement model
    ckpt_local = torch.load(ckpt_local)
    local_net.load_state_dict(ckpt_local)
    local_net = local_net.to(device)
    local_net.eval()

    source_frames, atlas_frames = load_videos(
        stylized, video_frames_reconstruction, device
    )
    
    frames_preds = []

    with torch.no_grad():
        for id in tqdm.tqdm(range(len(source_frames)), desc="Running refinement"):
            frame_content = source_frames[id]
            frame_style = atlas_frames[id]

            padder = InputPadder(frame_content.shape)
            frame_content, frame_style = padder.pad(frame_content, frame_style)

            frame_pred = filter_net(torch.cat([frame_content, frame_style], dim=1))
            
            ### local_net
            if id == 0:
                frame_o1 = frame_pred
                frame_o2 = frame_pred
                frame_p1 = frame_pred
            else:
                frame_p2 = frame_pred
                inputs = torch.cat((frame_p2, frame_o1, frame_p2, frame_p1), dim=1)
                output, _ = local_net(inputs, None)
                frame_o2 = frame_p2 + output
                frame_p1 = frame_p2
                frame_o1 = frame_o2

            frame_content, frame_style, frame_pred = (
                tensor2img(frame_content),
                tensor2img(frame_style),
                tensor2img(frame_pred),
            )
            
            frame_o2_np = tensor2img(frame_o2)
            
            frame_concat = np.concatenate([frame_style, frame_pred], axis=1)
            frames_preds.append(frame_o2_np)
            save_img(frame_o2_np, f"{output_folder}/{id:05d}.png")
            save_img(frame_concat, f"{output_folder_concat}/{id:05d}.png")

    return frames_preds


def tensor2img(img_t: torch.Tensor):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img

def save_img(img, filename):
    if img.ndim == 3:
        img = img[:, :, ::-1]  ### RGB to BGR

    ## clip to [0, 1]
    img = np.clip(img, 0, 1)

    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)

    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def load_videos(stylized: list[np.ndarray], reconstruction: np.ndarray, device):
    # stylized: [540, 960, 3] [0 255]
    # reconstruction: [540, 960, 3, 11] [0 1]
    if isinstance(reconstruction, np.ndarray):
        assert (
            len(stylized) == reconstruction.shape[-1]
        ), f"Number of frames mismatch. {len(stylized)=} {reconstruction.shape[-1]=}"
    elif isinstance(reconstruction, list):
        assert (len(stylized)) == len(
            reconstruction
        ), f"Number of frames mismatch. {len(stylized)=} {len(reconstruction)=}"

    source_frames: list[torch.Tensor] = []
    atlas_frames: list[torch.Tensor] = []

    h, w, c = stylized[0].shape

    for i in tqdm.tqdm(range(len(stylized)), desc="Loading videos"):
        src_np = cv2.cvtColor(stylized[i], cv2.COLOR_RGB2BGR) / 255.0
        if isinstance(reconstruction, np.ndarray):
            recon_np = reconstruction[:, :, :, i]
            recon_np = cv2.resize(recon_np, (w, h), cv2.INTER_LINEAR)
        elif isinstance(reconstruction, list):
            recon_np = cv2.cvtColor(reconstruction[i], cv2.COLOR_RGB2BGR) / 255.0

        recon_ts = (
            torch.from_numpy(recon_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device, dtype=torch.float)
        )
        src_ts = (
            torch.from_numpy(src_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device, dtype=torch.float)
        )

        source_frames.append(src_ts)
        atlas_frames.append(recon_ts)

    return source_frames, atlas_frames
