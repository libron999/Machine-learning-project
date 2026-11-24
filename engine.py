"""One alternating WGAN-GP training step, independently testable on CPU."""
import torch
from common import assert_finite, set_grad
from losses import content_loss, gradient_penalty, luminance


def training_step(generator, dir_, dvis, optimizers, ir, vis, cfg):
    og, oi, ov = optimizers
    generator.train()
    dir_.train()
    dvis.train()
    # No generator BN update during the detached discriminator-only pass.
    generator.eval()
    with torch.no_grad():
        detached = generator(ir, vis)
    generator.train()
    stats = {}
    for name, disc, opt, real, fake in (
            ('dir', dir_, oi, ir, luminance(detached)),
            ('dvis', dvis, ov, vis, detached)):
        set_grad(disc, True)
        values = []
        for _ in range(cfg['train']['d_steps']):
            opt.zero_grad(set_to_none=True)
            gp = gradient_penalty(disc, real, fake)
            loss = disc(fake).mean() - disc(real).mean() + cfg['loss']['gp'] * gp
            assert_finite(loss, name + ' loss')
            loss.backward()
            for parameter in disc.parameters():
                if parameter.grad is not None:
                    assert_finite(parameter.grad, name + ' gradient')
            opt.step()
            values.append(loss.detach().item())
        stats[name] = sum(values) / len(values)
        opt.zero_grad(set_to_none=True)
        set_grad(disc, False)
    og.zero_grad(set_to_none=True)
    fused = generator(ir, vis)
    content, parts = content_loss(fused, ir, vis, cfg['loss'])
    adv_ir = -dir_(luminance(fused)).mean()
    adv_vis = -dvis(fused).mean()
    loss = content + cfg['loss']['adv_ir'] * adv_ir + cfg['loss']['adv_vis'] * adv_vis
    assert_finite(loss, 'generator loss')
    loss.backward()
    for parameter in generator.parameters():
        if parameter.grad is not None:
            assert_finite(parameter.grad, 'generator gradient')
    og.step()
    set_grad(dir_, True)
    set_grad(dvis, True)
    stats.update({k: v.detach().item() for k, v in parts.items()})
    stats.update(g=loss.detach().item(), content=content.detach().item(),
                 adv_ir=adv_ir.detach().item(), adv_vis=adv_vis.detach().item())
    return stats
