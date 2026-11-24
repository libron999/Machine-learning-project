"""All image inputs are in [-1,1]; no channel broadcasting of IR into RGB."""
import torch
import torch.nn.functional as F


def luminance(rgb):
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError('luminance expects BCHW RGB')
    return (rgb * rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)


def chroma(rgb):
    # Centered Cb/Cr on the SAME [-1,1] image scale; no video-range offsets.
    y = luminance(rgb)
    return torch.cat((0.564 * (rgb[:, 2:3] - y), 0.713 * (rgb[:, :1] - y)), 1)


def content_loss(fused, ir, vis, cfg):
    y = luminance(fused)
    mse = F.mse_loss(y, ir)
    l1 = F.l1_loss(y, luminance(vis))
    color = F.l1_loss(chroma(fused), chroma(vis))
    total = mse + cfg['visible_l1'] * l1 + cfg['color'] * color
    return total, {'ir_mse': mse, 'vis_l1': l1, 'color_l1': color}


def gradient_penalty(discriminator, real, fake):
    if real.shape != fake.shape:
        raise ValueError('Gradient penalty inputs must have identical shapes')
    # Full precision is required for second-order gradients.
    with torch.autocast(device_type=real.device.type, enabled=False):
        alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
        interpolated = (alpha * real.detach().float() + (1-alpha) * fake.detach().float()).requires_grad_(True)
        scores = discriminator(interpolated)
        gradient = torch.autograd.grad(scores.sum(), interpolated, create_graph=True)[0]
        norms = torch.linalg.vector_norm(gradient.flatten(1), dim=1)
        return (norms - 1).square().mean()
