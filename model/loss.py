import torch
import torch.nn.functional as F
# local modules
from PerceptualSimilarity import models
from utils import loss


class combined_perceptual_loss():
    def __init__(self, weight=1.0, use_gpu=True):
        """
        Flow wrapper for perceptual_loss
        """
        self.loss = perceptual_loss(weight=1.0, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred_img, pred_flow, target_img, target_flow):
        """
        image is tensor of N x 2 x H x W, flow of N x 2 x H x W
        These are concatenated, as perceptualLoss expects N x 3 x H x W.
        """
        pred = torch.cat([pred_img, pred_flow], dim=1)
        target = torch.cat([target_img, target_flow], dim=1)
        dist = self.loss(pred, target, normalize=False)
        return dist * self.weight


class warping_flow_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = loss.warping_flow_loss
        self.weight = weight
        self.L0 = L0
        self.default_return = None

    def __call__(self, i, image1, flow):
        """
        flow is from image0 to image1 (reversed when passed to
        warping_flow_loss function)
        """
        loss = self.default_return if i < self.L0 else self.weight * self.loss(
                self.image0, image1, -flow)
        self.image0 = image1
        return loss


class voxel_warp_flow_loss():
    def __init__(self, weight=1.0):
        self.loss = loss.voxel_warping_flow_loss
        self.weight = weight

    def __call__(self, voxel, displacement, output_images=False):
        """
        Warp the voxel grid by the displacement map. Variance 
        of resulting image is loss
        """
        loss = self.loss(voxel, displacement, output_images)
        if output_images:
            loss = (self.weight * loss[0], loss[1])
        else:
            loss *= self.weight
        return loss


class flow_perceptual_loss():
    def __init__(self, weight=1.0, use_gpu=True):
        """
        Flow wrapper for perceptual_loss
        """
        self.loss = perceptual_loss(weight=1.0, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred, target):
        """
        pred and target are Tensors with shape N x 2 x H x W
        PerceptualLoss expects N x 3 x H x W.
        """
        dist_x = self.loss(pred[:, 0:1, :, :], target[:, 0:1, :, :], normalize=False)
        dist_y = self.loss(pred[:, 1:2, :, :], target[:, 1:2, :, :], normalize=False)
        return (dist_x + dist_y) / 2 * self.weight


class flow_l1_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


# keep for compatibility
flow_loss = flow_l1_loss


class perceptual_loss():
    def __init__(self, weight=1.0, net='alex', use_gpu=True):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model.forward(pred, target, normalize=normalize)
        return self.weight * dist.mean()


class l2_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class temporal_consistency_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = loss.temporal_consistency_loss
        self.weight = weight
        self.L0 = L0

    def __call__(self, i, image1, processed1, flow, output_images=False):
        """
        flow is from image0 to image1 (reversed when passed to
        temporal_consistency_loss function)
        """
        if i >= self.L0:
            loss = self.loss(self.image0, image1, self.processed0, processed1,
                             -flow, output_images=output_images)
            if output_images:
                loss = (self.weight * loss[0], loss[1])
            else:
                loss *= self.weight
        else:
            loss = None
        self.image0 = image1
        self.processed0 = processed1
        return loss
