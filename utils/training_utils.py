import cv2
import numpy as np
import torch
from torchvision import utils
# local modules
from utils.myutil import quick_norm


def make_tc_vis(tc_output):
    imgs0 = [tc_output['image1'], tc_output['image1'], tc_output['visibility_mask']]
    imgs1 = [tc_output['image0'], tc_output['image0_warped_to1'], tc_output['visibility_mask']]
    frames = []
    for imgs in [imgs0, imgs1]:
        imgs = [i[0, ...].expand(3, -1, -1) for i in imgs]
        frames.append(utils.make_grid(imgs, nrow=3))
    return torch.stack(frames, dim=0).unsqueeze(0)

def make_vw_vis(tc_output):
    event_preview = torch.sum(tc_output['voxel_grid'], dim=1, keepdim=True)[0, ...].expand(3, -1, -1)
    events_warped = tc_output['voxel_grid_warped'][0, ...].expand(3, -1, -1)
    frames = []
    frames.append(utils.make_grid([event_preview, events_warped], nrow=2))
    frames.append(utils.make_grid([events_warped, event_preview], nrow=2))
    return torch.stack(frames, dim=0).unsqueeze(0)

def make_flow_movie(event_previews, predicted_frames, groundtruth_frames, predicted_flows, groundtruth_flows):
    # event_previews: a list of [1 x 1 x H x W] event previews
    # predicted_frames: a list of [1 x 1 x H x W] predicted frames
    # flows: a list of [1 x 2 x H x W] predicted frames
    # for movie, we need to pass [1 x T x 1 x H x W] where T is the time dimension max_magnitude = 40
    if groundtruth_flows is None:
        groundtruth_flows = []
    max_magnitude = None
    movie_frames = []
    for i, flow in enumerate(predicted_flows):
        voxel = quick_norm(event_previews[i][0, ...]).expand(3, -1, -1)
        pred_frame = quick_norm(predicted_frames[i][0, ...]).expand(3, -1, -1)
        gt_frame = groundtruth_frames[i][0, ...].expand(3, -1, -1)
        pred_flow_rgb = flow2rgb(flow[0, 0, :, :], flow[0, 1, :, :], max_magnitude)
        blank = torch.zeros_like(gt_frame)
        imgs = [voxel, pred_frame, gt_frame, blank, pred_flow_rgb.float()]
        if groundtruth_flows:
            gt_flow = groundtruth_flows[i]
            gt_flow_rgb = flow2rgb(gt_flow[0, 0, :, :], gt_flow[0, 1, :, :], max_magnitude)
            imgs.append(gt_flow_rgb.float())
        movie_frame = utils.make_grid(imgs, nrow=3)
        movie_frames.append(movie_frame)
    return torch.stack(movie_frames, dim=0).unsqueeze(0)


def flush(summary_writer):
    for writer in summary_writer.all_writers.values():
        writer.flush()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_evenly_spaced_elements(num_elements, sequence_length):
    return [i * sequence_length // num_elements + sequence_length // (2 * num_elements) for i in range(num_elements)]


def flow2bgr_np(disp_x, disp_y, max_magnitude=None):
    """
    Convert an optic flow tensor to an RGB color map for visualization
    Code adapted from: https://github.com/ClementPinard/FlowNetPytorch/blob/master/main.py#L339

    :param disp_x: a [H x W] NumPy array containing the X displacement
    :param disp_x: a [H x W] NumPy array containing the Y displacement
    :returns bgr: a [H x W x 3] NumPy array containing a color-coded representation of the flow [0, 255]
    """
    assert(disp_x.shape == disp_y.shape)
    H, W = disp_x.shape

    # X, Y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))

    # flow_x = (X - disp_x) * float(W) / 2
    # flow_y = (Y - disp_y) * float(H) / 2
    # magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    # magnitude, angle = cv2.cartToPolar(disp_x, disp_y)

    # follow alex zhu color convention https://github.com/daniilidis-group/EV-FlowNet

    flows = np.stack((disp_x, disp_y), axis=2)
    magnitude = np.linalg.norm(flows, axis=2)

    angle = np.arctan2(disp_y, disp_x)
    angle += np.pi
    angle *= 180. / np.pi / 2.
    angle = angle.astype(np.uint8)

    if max_magnitude is None:
        v = np.zeros(magnitude.shape, dtype=np.uint8)
        cv2.normalize(src=magnitude, dst=v, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        v = np.clip(255.0 * magnitude / max_magnitude, 0, 255)
        v = v.astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = angle
    hsv[..., 2] = v
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def flow2rgb(disp_x, disp_y, max_magnitude=None):
    return flow2bgr(disp_x, disp_y, max_magnitude)[[2, 1, 0], ...]


def flow2bgr(disp_x, disp_y, max_magnitude=None):
    device = disp_x.device
    bgr = flow2bgr_np(disp_x.cpu().numpy(), disp_y.cpu().numpy(), max_magnitude)
    bgr = bgr.astype(float) / 255
    return torch.tensor(bgr).permute(2, 0, 1).to(device)  # 3 x H x W


def make_movie(event_previews, predicted_frames, groundtruth_frames):
    # event_previews: a list of [1 x 1 x H x W] event previews
    # predicted_frames: a list of [1 x 1 x H x W] predicted frames
    # for movie, we need to pass [1 x T x 1 x H x W] where T is the time dimension

    video_tensor = None
    for i in torch.arange(len(event_previews)):
        voxel = quick_norm(event_previews[i])
        predicted_frame = quick_norm(predicted_frames[i])
        movie_frame = torch.cat([voxel,
                                 predicted_frame,
                                 groundtruth_frames[i]],
                                 dim=-1)
        movie_frame.unsqueeze_(dim=0)
        video_tensor = movie_frame if video_tensor is None else \
            torch.cat((video_tensor, movie_frame), dim=1)
    return video_tensor
