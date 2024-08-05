import json
import os

import cv2
import imageio
import numpy as np
import torch
import tqdm

from consistency.implicit_neural_networks import IMLP
from ezsynth.utils.flow_utils.OpticalFlow import RAFT_flow


def deflicker(
    cfg_path: str,
    checkpoint_path: str,
    img_frs_seq: list[np.ndarray],
    rafter: RAFT_flow,
    iters_num=10001,
    down_scale=1,
    results_folder="output_reconstruct",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(cfg_path) as f:
        cfg = json.load(f)

    num_frames = len(img_frs_seq)

    resx = img_frs_seq[0].shape[1] // down_scale
    resy = img_frs_seq[0].shape[0] // down_scale
    larger_dim = np.maximum(resx, resy)

    frames, frames_dx, frames_dy, flow_fwds, flow_bwds, mask_fwds, mask_bwds = (
        compute_both_flow(img_frs_seq, rafter, resx, resy, num_frames)
    )

    model_F_mapping1 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=cfg["number_of_channels_mapping1"],
        use_positional=cfg["use_positional_encoding_mapping1"],
        positional_dim=cfg["number_of_positional_encoding_mapping1"],
        num_layers=cfg["number_of_layers_mapping1"],
        skip_layers=[],
    ).to(device)

    model_F_atlas = IMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=cfg["number_of_channels_atlas"],
        use_positional=True,
        positional_dim=cfg["positional_encoding_num_atlas"],
        num_layers=cfg["number_of_layers_atlas"],
        skip_layers=[4, 7],
    ).to(device)

    optimizer_all = torch.optim.Adam(
        [
            {"params": list(model_F_mapping1.parameters())},
            {"params": list(model_F_atlas.parameters())},
        ],
        lr=0.0001,
    )

    start_iteration = 0
    # init_file = torch.load(checkpoint_path)
    # model_F_atlas.load_state_dict(init_file["F_atlas_state_dict"])
    # model_F_mapping1.load_state_dict(init_file["model_F_mapping1_state_dict"])
    # optimizer_all.load_state_dict(init_file["optimizer_all_state_dict"])
    # start_iteration = init_file["iteration"]

    model_F_mapping1 = pre_train_mapping(
        model_F_mapping1,
        num_frames,
        cfg["uv_mapping_scale"],
        resx=resx,
        resy=resy,
        larger_dim=larger_dim,
        device=device,
        pretrain_iters=cfg['pretrain_iter_number'],
    )

    model_F_atlas.train()
    model_F_mapping1.train()

    jif_all = get_tuples(num_frames, frames)

    loss_val = 0.0

    # Start training!
    for i in (pbar := tqdm.tqdm(range(start_iteration, iters_num))):
        pbar.set_description(
            f"Training reconstruct model. Last loss: {loss_val:.3f}"
        )
        if i > cfg["stop_global_rigidity"]:
            global_rigidity_coeff_fg = 0
        else:
            global_rigidity_coeff_fg = cfg['global_rigidity_coeff_fg']

        # randomly choose indices for the current batch
        inds_foreground = torch.randint(
            jif_all.shape[1], (np.int64(cfg["samples_batch"] * 1.0), 1)
        )

        jif_current = jif_all[:, inds_foreground]  # size (3, batch, 1)

        rgb_current = (
            frames[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]]
            .squeeze(1)
            .to(device)
        )

        # normalize coordinates to be in [-1,1]
        xyt_current = torch.cat(
            (
                jif_current[0, :] / (larger_dim / 2) - 1,
                jif_current[1, :] / (larger_dim / 2) - 1,
                jif_current[2, :] / (num_frames / 2.0) - 1,
            ),
            dim=1,
        ).to(device)  # size (batch, 3)

        # get the atlas UV coordinates from the two mapping networks;
        uv_foreground1 = model_F_mapping1(xyt_current)

        # direct set alpha to one
        alpha = torch.ones(cfg["samples_batch"], 1).to(device)

        # Sample atlas values. Foreground colors are sampled from [0,1]x[0,1] and background colors are sampled from [-1,0]x[-1,0]
        # Note that the original [u,v] coorinates are in [-1,1]x[-1,1] for both networks
        rgb_output1 = (model_F_atlas(uv_foreground1 * 0.5 + 0.5) + 1.0) * 0.5
        # Reconstruct final colors from the two layers (using alpha)
        rgb_output_foreground = rgb_output1

        # if cfg["use_gradient_loss"]:
        #     gradient_loss = get_gradient_loss_single(
        #         frames_dx,
        #         frames_dy,
        #         jif_current,
        #         model_F_mapping1,
        #         model_F_atlas,
        #         rgb_output_foreground,
        #         device,
        #         resx,
        #         num_frames,
        #     )
        # else:
        #     gradient_loss = 0.0
        
        gradient_loss = 0.0 
        
        rgb_loss = (
            torch.norm(rgb_output_foreground - rgb_current, dim=1) ** 2
        ).mean()

        rigidity_loss1 = get_rigidity_loss(
            jif_current,
            cfg["derivative_amount"],
            larger_dim,
            num_frames,
            model_F_mapping1,
            uv_foreground1,
            device,
            uv_mapping_scale=cfg["uv_mapping_scale"],
        )

        if cfg["include_global_rigidity_loss"] and i <= cfg["stop_global_rigidity"]:
            global_rigidity_loss1 = get_rigidity_loss(
                jif_current,
                cfg["global_rigidity_derivative_amount_fg"],
                larger_dim,
                num_frames,
                model_F_mapping1,
                uv_foreground1,
                device,
                uv_mapping_scale=cfg["uv_mapping_scale"],
            )

        # flow_loss1 = get_optical_flow_loss(
        #     jif_current,
        #     uv_foreground1,
        #     flow_fwds,
        #     flow_bwds,
        #     larger_dim,
        #     num_frames,
        #     model_F_mapping1,
        #     mask_fwds,
        #     mask_bwds,
        #     cfg["uv_mapping_scale"],
        #     device,
        #     use_alpha=True,
        #     alpha=alpha,
        # )
        
        flow_loss1 = 0

        if cfg["include_global_rigidity_loss"] and i <= cfg["stop_global_rigidity"]:
            loss = (
                cfg["rigidity_coeff"] * (rigidity_loss1)
                + global_rigidity_coeff_fg * global_rigidity_loss1
                + rgb_loss * cfg["rgb_coeff"]
                + cfg["optical_flow_coeff"] * (flow_loss1)
                + gradient_loss * cfg["gradient_loss_coeff"]
            )
        else:
            loss = (
                cfg["rigidity_coeff"] * (rigidity_loss1)
                + rgb_loss * cfg["rgb_coeff"]
                + cfg["optical_flow_coeff"] * (flow_loss1)
                + gradient_loss * cfg["gradient_loss_coeff"]
            )

        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()
        
        loss_val = loss.item()
    print()
    print("Reconstructing")
    video_frames_reconstruction = reconstruct(
        resx, resy, larger_dim, num_frames, model_F_atlas, model_F_mapping1, device
    )

    print("Saving")
    for i in range(num_frames):
        # save image
        save_img_path = os.path.join(results_folder, "output", "%05d.png" % i)
        save_image(save_img_path, video_frames_reconstruction, i)

def save_image(save_img_path, video_frames_reconstruction, i):
    
    # Ensure the directory exists
    dir_name = os.path.dirname(save_img_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save the image
    imageio.imwrite(
        save_img_path,
        (video_frames_reconstruction[:, :, :, i] * 255).astype(np.uint8)
    )

def get_tuples(num_frames, frames):
    # video_frames shape: (resy, resx, 3, num_frames)
    jif_all = []
    for f in range(num_frames):
        mask = (frames[:, :, :, f] > -1).any(dim=2)
        relis, reljs = torch.where(mask > 0.5)
        jif_all.append(torch.stack((reljs, relis, f * torch.ones_like(reljs))))
    return torch.cat(jif_all, dim=1)


def reconstruct(
    resx, resy, larger_dim, num_frames, model_F_atlas, model_F_mapping1, device
):
    video_frames_reconstruction = np.zeros((resy, resx, 3, num_frames))

    for f in tqdm.tqdm(range(num_frames), desc="Reconstructing"):
        relis_i, reljs_i = torch.where(torch.ones(resy, resx) > 0)

        # split the coordinates of the entire image such that no more than 100k coordinates in each batch
        relisa = np.array_split(relis_i.numpy(), np.ceil(relis_i.shape[0] / 100000))
        reljsa = np.array_split(reljs_i.numpy(), np.ceil(relis_i.shape[0] / 100000))

        for i in range(len(relisa)):
            relis = torch.from_numpy(relisa[i]).unsqueeze(1) / (larger_dim / 2) - 1
            reljs = torch.from_numpy(reljsa[i]).unsqueeze(1) / (larger_dim / 2) - 1

            # Map video indices to uv coordinates using the two mapping networks:
            uv_temp1 = model_F_mapping1(
                torch.cat(
                    (
                        reljs,
                        relis,
                        (f / (num_frames / 2.0) - 1) * torch.ones_like(relis),
                    ),
                    dim=1,
                ).to(device)
            )

            # Sample RGB values from the atlas:
            rgb_current = model_F_atlas(uv_temp1 * 0.5 + 0.5)
            rgb_current = (rgb_current + 1) * 0.5

            video_frames_reconstruction[relisa[i], reljsa[i], :, f] = (
                rgb_current.detach().cpu().numpy()
            )

    return video_frames_reconstruction


def compute_both_flow(
    img_frs_seq: list[np.ndarray],
    rafter: RAFT_flow,
    resx: int,
    resy: int,
    num_frames: int,
):
    frames = torch.zeros((resy, resx, 3, num_frames))
    frames_dx = torch.zeros((resy, resx, 3, num_frames))
    frames_dy = torch.zeros((resy, resx, 3, num_frames))

    flow_fwds = torch.zeros((resy, resx, 2, num_frames))
    flow_bwds = torch.zeros((resy, resx, 2, num_frames))
    mask_fwds = torch.zeros((resy, resx, num_frames))
    mask_bwds = torch.zeros((resy, resx, num_frames))

    do_resize = resx != img_frs_seq[0].shape[1] or resy != img_frs_seq[0].shape[0]

    for i in tqdm.tqdm(range(num_frames), desc="Compute both flows"):
        im = (
            cv2.resize(img_frs_seq[i], (resx, resy), interpolation=cv2.INTER_LINEAR)
            if do_resize
            else img_frs_seq[i]
        )
        frames[:, :, :, i] = torch.from_numpy(im)

        if i < num_frames - 1:
            im_next = (
                cv2.resize(
                    img_frs_seq[i + 1], (resx, resy), interpolation=cv2.INTER_LINEAR
                )
                if do_resize
                else img_frs_seq[i + 1]
            )

            flow_fwd = rafter._compute_flow(im, im_next)
            flow_bwd = rafter._compute_flow(im_next, im)
            mask_fwd, mask_bwd = compute_consistency_mask(flow_fwd, flow_bwd)

            flow_fwds[:, :, :, i] = torch.from_numpy(flow_fwd)
            flow_bwds[:, :, :, i] = torch.from_numpy(flow_bwd)
            mask_fwds[:, :, i] = mask_fwd
            mask_bwds[:, :, i] = mask_bwd

    # Compute frame differences
    frames_dx = frames[:, 1:, :, :] - frames[:, :-1, :, :]
    frames_dy = frames[1:, :, :, :] - frames[:-1, :, :, :]

    return frames, frames_dx, frames_dy, flow_fwds, flow_bwds, mask_fwds, mask_bwds


def compute_consistency_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray):
    mask_flow_fwd = compute_consistency(flow_fwd, flow_bwd) < 1.0
    mask_flow_bwd = compute_consistency(flow_bwd, flow_fwd) < 1.0
    return torch.from_numpy(mask_flow_fwd), torch.from_numpy(mask_flow_bwd)


def compute_consistency(flow_fwd: np.ndarray, flow_bwd: np.ndarray):
    wflow_bwd = warp_flow(flow_bwd, flow_fwd)
    diff = flow_fwd + wflow_bwd
    diff = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2) ** 0.5
    return diff


def warp_flow(img: np.ndarray, flow: np.ndarray):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def resize_flow(flow: np.ndarray, newh: int, neww: int):
    oldh, oldw = flow.shape[0:2]
    flow = cv2.resize(flow, (neww, newh), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= newh / oldh
    flow[:, :, 1] *= neww / oldw
    return flow


# calculating the gradient loss as defined by Eq.7 in the paper for only one mapping network
def get_gradient_loss_single(
    video_frames_dx,
    video_frames_dy,
    jif_current,
    model_F_mapping1,
    model_F_atlas,
    rgb_output_foreground,
    device,
    resx,
    number_of_frames,
):
    xplus1yt_foreground = torch.cat(
        (
            (jif_current[0, :] + 1) / (resx / 2) - 1,
            jif_current[1, :] / (resx / 2) - 1,
            jif_current[2, :] / (number_of_frames / 2.0) - 1,
        ),
        dim=1,
    ).to(device)

    xyplus1t_foreground = torch.cat(
        (
            (jif_current[0, :]) / (resx / 2) - 1,
            (jif_current[1, :] + 1) / (resx / 2) - 1,
            jif_current[2, :] / (number_of_frames / 2.0) - 1,
        ),
        dim=1,
    ).to(device)

    # precomputed discrete derivative with respect to x,y direction
    rgb_dx_gt = (
        video_frames_dx[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]]
        .squeeze(1)
        .to(device)
    )
    rgb_dy_gt = (
        video_frames_dy[jif_current[1, :], jif_current[0, :], :, jif_current[2, :]]
        .squeeze(1)
        .to(device)
    )

    # uv coordinates for locations with offsets of 1 pixel
    uv_foreground1_xyplus1t = model_F_mapping1(xyplus1t_foreground)
    uv_foreground1_xplus1yt = model_F_mapping1(xplus1yt_foreground)

    # The RGB values (from the 2 layers) for locations with offsets of 1 pixel
    rgb_output1_xyplus1t = (
        model_F_atlas(uv_foreground1_xyplus1t * 0.5 + 0.5) + 1.0
    ) * 0.5
    rgb_output1_xplus1yt = (
        model_F_atlas(uv_foreground1_xplus1yt * 0.5 + 0.5) + 1.0
    ) * 0.5

    # Reconstructed RGB values:
    rgb_output_foreground_xyplus1t = rgb_output1_xyplus1t
    rgb_output_foreground_xplus1yt = rgb_output1_xplus1yt

    # Use reconstructed RGB values for computing derivatives:
    rgb_dx_output = rgb_output_foreground_xplus1yt - rgb_output_foreground
    rgb_dy_output = rgb_output_foreground_xyplus1t - rgb_output_foreground
    gradient_loss = torch.mean(
        (rgb_dx_gt - rgb_dx_output).norm(dim=1) ** 2
        + (rgb_dy_gt - rgb_dy_output).norm(dim=1) ** 2
    )
    return gradient_loss


# get rigidity loss as defined in Eq. 9 in the paper
def get_rigidity_loss(
    jif_foreground,
    derivative_amount,
    resx,
    number_of_frames,
    model_F_mapping,
    uv_foreground,
    device,
    uv_mapping_scale=1.0,
    return_all=False,
):
    # concatenating (x,y-derivative_amount,t) and (x-derivative_amount,y,t) to get xyt_p:
    is_patch = (
        torch.cat((jif_foreground[1, :] - derivative_amount, jif_foreground[1, :]))
        / (resx / 2)
        - 1
    )
    js_patch = (
        torch.cat((jif_foreground[0, :], jif_foreground[0, :] - derivative_amount))
        / (resx / 2)
        - 1
    )
    fs_patch = (
        torch.cat((jif_foreground[2, :], jif_foreground[2, :]))
        / (number_of_frames / 2.0)
        - 1
    )
    xyt_p = torch.cat((js_patch, is_patch, fs_patch), dim=1).to(device)

    uv_p = model_F_mapping(xyt_p)
    u_p = uv_p[:, 0].view(
        2, -1
    )  # u_p[0,:]= u(x,y-derivative_amount,t).  u_p[1,:]= u(x-derivative_amount,y,t)
    v_p = uv_p[:, 1].view(
        2, -1
    )  # v_p[0,:]= u(x,y-derivative_amount,t).  v_p[1,:]= v(x-derivative_amount,y,t)

    u_p_d_ = (
        uv_foreground[:, 0].unsqueeze(0) - u_p
    )  # u_p_d_[0,:]=u(x,y,t)-u(x,y-derivative_amount,t)   u_p_d_[1,:]= u(x,y,t)-u(x-derivative_amount,y,t).
    v_p_d_ = (
        uv_foreground[:, 1].unsqueeze(0) - v_p
    )  # v_p_d_[0,:]=u(x,y,t)-v(x,y-derivative_amount,t).  v_p_d_[1,:]= u(x,y,t)-v(x-derivative_amount,y,t).

    # to match units: 1 in uv coordinates is resx/2 in image space.
    du_dx = u_p_d_[1, :] * resx / 2
    du_dy = u_p_d_[0, :] * resx / 2
    dv_dy = v_p_d_[0, :] * resx / 2
    dv_dx = v_p_d_[1, :] * resx / 2

    jacobians = torch.cat(
        (
            torch.cat(
                (du_dx.unsqueeze(-1).unsqueeze(-1), du_dy.unsqueeze(-1).unsqueeze(-1)),
                dim=2,
            ),
            torch.cat(
                (dv_dx.unsqueeze(-1).unsqueeze(-1), dv_dy.unsqueeze(-1).unsqueeze(-1)),
                dim=2,
            ),
        ),
        dim=1,
    )
    jacobians = jacobians / uv_mapping_scale
    jacobians = jacobians / derivative_amount

    # Apply a loss to constrain the Jacobian to be a rotation matrix as much as possible
    JtJ = torch.matmul(jacobians.transpose(1, 2), jacobians)

    a = JtJ[:, 0, 0] + 0.001
    b = JtJ[:, 0, 1]
    c = JtJ[:, 1, 0]
    d = JtJ[:, 1, 1] + 0.001

    JTJinv = torch.zeros_like(jacobians).to(device)
    JTJinv[:, 0, 0] = d
    JTJinv[:, 0, 1] = -b
    JTJinv[:, 1, 0] = -c
    JTJinv[:, 1, 1] = a
    JTJinv = JTJinv / ((a * d - b * c).unsqueeze(-1).unsqueeze(-1))

    # See Equation (9) in the paper:
    rigidity_loss = (JtJ**2).sum(1).sum(1).sqrt() + (JTJinv**2).sum(1).sum(1).sqrt()

    if return_all:
        return rigidity_loss
    else:
        return rigidity_loss.mean()


# Compute optical flow loss (Eq. 11 in the paper)
def get_optical_flow_loss(
    jif_foreground,
    uv_foreground,
    optical_flows_reverse,
    optical_flows_reverse_mask,
    resx,
    number_of_frames,
    model_F_mapping,
    optical_flows,
    optical_flows_mask,
    uv_mapping_scale,
    device,
    use_alpha=False,
    alpha=1.0,
):
    # Forward flow:
    (
        uv_foreground_forward_relevant,
        xyt_foreground_forward_should_match,
        relevant_batch_indices_forward,
    ) = get_corresponding_flow_matches(
        jif_foreground,
        optical_flows_mask,
        optical_flows,
        resx,
        number_of_frames,
        True,
        uv_foreground,
    )
    uv_foreground_forward_should_match = model_F_mapping(
        xyt_foreground_forward_should_match.to(device)
    )
    loss_flow_next = (
        (uv_foreground_forward_should_match - uv_foreground_forward_relevant).norm(
            dim=1
        )
        * resx
        / (2 * uv_mapping_scale)
    )

    # Backward flow:
    (
        uv_foreground_backward_relevant,
        xyt_foreground_backward_should_match,
        relevant_batch_indices_backward,
    ) = get_corresponding_flow_matches(
        jif_foreground,
        optical_flows_reverse_mask,
        optical_flows_reverse,
        resx,
        number_of_frames,
        False,
        uv_foreground,
    )
    uv_foreground_backward_should_match = model_F_mapping(
        xyt_foreground_backward_should_match.to(device)
    )
    loss_flow_prev = (
        (uv_foreground_backward_should_match - uv_foreground_backward_relevant).norm(
            dim=1
        )
        * resx
        / (2 * uv_mapping_scale)
    )

    if use_alpha:
        flow_loss = (
            loss_flow_prev * alpha[relevant_batch_indices_backward].squeeze()
        ).mean() * 0.5 + (
            loss_flow_next * alpha[relevant_batch_indices_forward].squeeze()
        ).mean() * 0.5
    else:
        flow_loss = (loss_flow_prev).mean() * 0.5 + (loss_flow_next).mean() * 0.5

    return flow_loss


# A helper function for get_optical_flow_loss to return matching points according to the optical flow
def get_corresponding_flow_matches(
    jif_foreground,
    optical_flows_mask,
    optical_flows,
    resx,
    number_of_frames,
    is_forward,
    uv_foreground,
    use_uv=True,
):
    batch_forward_mask = torch.where(
        optical_flows_mask[
            jif_foreground[1, :].squeeze(),
            jif_foreground[0, :].squeeze(),
            jif_foreground[2, :].squeeze(),
            :,
        ]
    )
    forward_frames_amount = 2 ** batch_forward_mask[1]
    relevant_batch_indices = batch_forward_mask[0]
    jif_foreground_forward_relevant = jif_foreground[:, relevant_batch_indices, 0]
    forward_flows_for_loss = optical_flows[
        jif_foreground_forward_relevant[1],
        jif_foreground_forward_relevant[0],
        :,
        jif_foreground_forward_relevant[2],
        batch_forward_mask[1],
    ]

    if is_forward:
        jif_foreground_forward_should_match = torch.stack(
            (
                jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
                jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
                jif_foreground_forward_relevant[2] + forward_frames_amount,
            )
        )
    else:
        jif_foreground_forward_should_match = torch.stack(
            (
                jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
                jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
                jif_foreground_forward_relevant[2] - forward_frames_amount,
            )
        )

    xyt_foreground_forward_should_match = torch.stack(
        (
            jif_foreground_forward_should_match[0] / (resx / 2) - 1,
            jif_foreground_forward_should_match[1] / (resx / 2) - 1,
            jif_foreground_forward_should_match[2] / (number_of_frames / 2) - 1,
        )
    ).T
    if use_uv:
        uv_foreground_forward_relevant = uv_foreground[batch_forward_mask[0]]
        return (
            uv_foreground_forward_relevant,
            xyt_foreground_forward_should_match,
            relevant_batch_indices,
        )
    else:
        return xyt_foreground_forward_should_match, relevant_batch_indices


# See explanation in the paper, appendix A (Second paragraph)
def pre_train_mapping(
    model_F_mapping,
    frames_num,
    uv_mapping_scale,
    resx,
    resy,
    larger_dim,
    device,
    pretrain_iters=100,
):
    optimizer_mapping = torch.optim.Adam(model_F_mapping.parameters(), lr=0.0001)
    print("pre-training")
    for i in tqdm.tqdm(range(pretrain_iters), desc="Pretraining"):
        for f in range(frames_num):
            i_s_int = torch.randint(resy, (np.int64(10000), 1))
            j_s_int = torch.randint(resx, (np.int64(10000), 1))

            i_s = i_s_int / (larger_dim / 2) - 1
            j_s = j_s_int / (larger_dim / 2) - 1

            xyt = torch.cat(
                (j_s, i_s, (f / (frames_num / 2.0) - 1) * torch.ones_like(i_s)), dim=1
            ).to(device)
            uv_temp = model_F_mapping(xyt)

            model_F_mapping.zero_grad()

            loss = (xyt[:, :2] * uv_mapping_scale - uv_temp).norm(dim=1).mean()
            # print(f"pre-train loss: {loss.item()}")
            loss.backward()
            optimizer_mapping.step()
    return model_F_mapping
