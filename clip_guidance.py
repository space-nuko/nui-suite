import torch
from torch.nn import functional as F
import comfy.model_management
import comfy.samplers
import math
import numpy as np
from torchvision import transforms
from resize_right import resize
from einops import rearrange
import clip
import torchviz

from comfy.sample import broadcast_cond, load_additional_models, cleanup_additional_models, prepare_mask
import comfy.k_diffusion as k_diffusion
from comfy.k_diffusion import external as k_diffusion_external


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, vae, clip, clip_vision, clip_prompt, clip_scale, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        pbar.update_absolute(step + 1, total_steps)

    samples = sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, vae,
                     clip, clip_vision, clip_prompt, clip_scale,
                     denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                     force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def differentiable_decode_first_stage(vae, z, predict_cids=False, force_not_quantize=False):
    if predict_cids:
        if z.dim() == 4:
            z = torch.argmax(z.exp(), dim=1).long()
        z = vae.first_stage_model.quantize.get_codebook_entry(z, shape=None)
        z = rearrange(z, 'b h w c -> b c h w').contiguous()

    z = 1. / vae.scale_factor * z
    return vae.first_stage_model.decode(z)


def decode(vae, samples_in):
    comfy.model_management.unload_model()
    vae.first_stage_model = vae.first_stage_model.to(vae.device)

    free_memory = comfy.model_management.get_free_memory(vae.device)
    batch_number = int((free_memory * 0.7) / (2562 * samples_in.shape[2] * samples_in.shape[3] * 64))
    batch_number = max(1, batch_number)

    pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device=vae.device)
    for x in range(0, samples_in.shape[0], batch_number):
        samples = samples_in[x:x+batch_number].to(vae.device)
        pixel_samples[x:x+batch_number] = torch.clamp((vae.first_stage_model.decode(1. / vae.scale_factor * samples) + 1.0) / 2.0, min=0.0, max=1.0)

    vae.first_stage_model = vae.first_stage_model.cpu()
    pixel_samples = pixel_samples.movedim(1,-1)
    return pixel_samples


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def decode(vae, samples_in):
    comfy.model_management.unload_model()
    vae.first_stage_model = vae.first_stage_model.to(vae.device)
    pxsmps = []
    smps = []

    free_memory = comfy.model_management.get_free_memory(vae.device)
    batch_number = int((free_memory * 0.7) / (2562 * samples_in.shape[2] * samples_in.shape[3] * 64))
    batch_number = max(1, batch_number)

    # pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
    for x in range(0, samples_in.shape[0], batch_number):
        samples = samples_in[x:x+batch_number].to(vae.device)
        px = torch.clamp((vae.first_stage_model.decode(1. / vae.scale_factor * samples) + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        pxsmps.append(px.cpu().float().movedim(1, -1))
        smps.append(samples)

    # vae.first_stage_model = vae.first_stage_model.cpu()
    # pixel_samples = pixel_samples.cpu().movedim(1,-1)
    return (pxsmps, smps)


class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, name='ViT-L/14@336px', device='cpu'):
        super().__init__()
        self.model = clip.load(name, device=device)[0].eval().requires_grad_(False)
        self.normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                              std=(0.26862954, 0.26130258, 0.27577711))
        self.size = (self.model.visual.input_resolution, self.model.visual.input_resolution)

    def forward(self, x):
        if x.shape[2:4] != self.size:
            x = resize(x.add(1).div(2), out_shape=self.size, pad_mode='reflect').clamp(0, 1)
        x = self.normalize(x)
        x = self.model.encode_image(x).float()
        x = F.normalize(x) * x.shape[1] ** 0.5
        return x


class CLIPGuidedNoisePredictor(torch.nn.Module):
    def __init__(self, pred, vae, clip, clip_vision, clip_target_embed, clip_scale, device):
        super().__init__()
        self.inner_pred = pred
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.clip_target_embed = clip_target_embed
        self.clip_scale = clip_scale
        clip_size = clip_vision.model.config.image_size
        self.clip_size = (clip_size, clip_size)
        self.clip_normalize = transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711))
        self.alphas_cumprod = self.inner_pred.alphas_cumprod
        cutn = 4
        self.make_cutouts = MakeCutouts(clip_size, cutn)
        self.clip2 = CLIPFeatureExtractor("ViT-L/14@336px", device)

    def apply_model(self, x, timestep, cond, uncond, cond_scale, cond_concat=None, model_options={}):
        print("apply_model")
        print(x.grad_fn)
        with torch.enable_grad():
            device = comfy.model_management.get_torch_device()
            x = x.detach().requires_grad_(True).to(device)
            print("================================")
            denoised = self.inner_pred.apply_model(x, timestep, cond, uncond, cond_scale, cond_concat=cond_concat, model_options=model_options).requires_grad_(True)
            print(x.grad_fn)
            print(denoised.grad_fn)
            print("+++++++++++++++++++++++++++++++")

            cond_grad = self.cond_fn(x, denoised=denoised, target_embed=self.clip_target_embed).detach()

            ndim = x.ndim
            del x
            cond_denoised = denoised.detach() + cond_grad * k_diffusion.utils.append_dims(timestep**2, ndim)

            return cond_denoised

    def cond_fn(self, x, denoised, target_embed):
        device = denoised.device
        pxsmps, smps = decode(self.vae, denoised)
        decoded = pxsmps[0].to(device)
        x_in = smps[0]

        print("1________________")
        print(x_in.shape)
        print(x_in.grad_fn)
        print(decoded.shape)
        print(decoded.grad_fn)
        print(denoised.get_device())
        print(decoded.get_device())
        print("2________________")

        # import torchviz
        # dot = torchviz.make_dot(decoded, params=dict(self.inner_pred.inner_model.named_parameters()))
        # with open("dot_decoded.dot", "w") as f:
        #     f.write(str(dot))
        # dot = torchviz.make_dot(denoised, params=dict(self.inner_pred.inner_model.named_parameters()))
        # with open("dot_denoised.dot", "w") as f:
        #     f.write(str(dot))
        # dot = torchviz.make_dot(x_in, params=dict(self.inner_pred.inner_model.named_parameters()))
        # with open("dot_x_in.dot", "w") as f:
        #     f.write(str(dot))

        del denoised
        renormalized = decoded.add(1).div(2)
        del decoded
        # # if self.clip_augmentations:
        # #     # this particular approach to augmentation crashes on MPS (Metal Performance Shaders, macOS), so we transfer to CPU (for now)
        # #     # :27:11: error: invalid input tensor shapes, indices shape and updates shape must be equal
        # #     # -:27:11: note: see current operation: %25 = "mps.scatter_along_axis"(%23, %arg3, %24, %1) {mode = 6 : i32} : (tensor<786432xf32>, tensor<512xf32>, tensor<262144xi32>, tensor<i32>) -> tensor<786432xf32>
        # #     # TODO: this approach (from k-diffusion example) produces just the one augmentation,
        # #     #       whereas diffusers approach is to use many and sum their losses. should we?
        # #     renormalized = self.aug(renormalized.cpu()).to(device) if device.type == 'mps' else self.aug(renormalized)
        clamped = renormalized.clamp(0, 1)
        clamped = rearrange(clamped, 'b h w c -> b c h w')
        del renormalized

        # cutouts = self.make_cutouts(decoded)

        # image_embed = self.get_image_embed(clamped)
        print(x.get_device())
        print(x_in.get_device())
        print(clamped.get_device())
        print(target_embed.get_device())
        image_embed = self.clip2(clamped)
        print(image_embed.get_device())

        # dot = torchviz.make_dot(image_embed, params=dict(self.clip_vision.model.named_parameters()))
        # with open("dot_image_embed.dot", "w") as f:
        #     f.write(str(dot))

        # del clamped
        # TODO: does this do the right thing for multi-sample?
        # TODO: do we want .mean() here or .sum()? or both?
        #       k-diffusion example used just .sum(), but k-diff was single-aug. maybe that was for multi-sample?
        #       whereas diffusers uses .mean() (this seemed to be over a single number, but maybe when you have multiple samples it becomes the mean of the loss over your n samples?),
        #       then uses sum() (which for multi-aug would sum the losses of each aug)
        print("---------incoming-------------------------")
        print(image_embed.shape)
        print(image_embed.grad_fn)
        print(target_embed.shape)
        print(target_embed.grad_fn)
        loss = spherical_dist_loss(target_embed, image_embed).sum() * self.clip_scale
        del image_embed
        # TODO: does this do the right thing for multi-sample?
        print("loss")
        print(loss)
        print(loss.shape)
        print(loss.grad_fn)
        print(loss.get_device())
        print("x")
        print(x_in.shape)
        print(x_in.grad_fn)
        print(x_in.get_device())
        print("------------------")
        from pprint import pp
        dot = torchviz.make_dot(x, params=dict(self.inner_pred.inner_model.named_parameters()))
        with open("dot_x.dot", "w") as f:
            f.write(str(dot))
        dot = torchviz.make_dot(x_in, params=dict(self.inner_pred.inner_model.named_parameters()))
        with open("dot_x_in.dot", "w") as f:
            f.write(str(dot))
        dot = torchviz.make_dot(loss, params=dict(self.inner_pred.inner_model.named_parameters()))
        with open("dot_loss.dot", "w") as f:
            f.write(str(dot))
        d= comfy.model_management.get_torch_device()
        grad = -torch.autograd.grad(loss.to(d), x_in.to(d))[0]

        return grad

    def get_image_embed(self, x):
        print(x.shape)
        print(self.clip_size)
        if x.shape[2:4] != self.clip_size:
            x = transforms.Resize(self.clip_size)(x)
            # x = resize(x.add(1).div(2), out_shape=self.clip_size, pad_mode='reflect').clamp(0, 1)
        # x = self.normalize(x)
        # x = self.clip_vision.encode_image(x).image_embeds.float()
        # x = F.normalize(x) * x.shape[1] ** 0.5
        x = self.clip_normalize(x)
        x = self.clip_vision.model(x)
        return x


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, vae, clip, clip_vision, clip_prompt, clip_scale, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False):
    device = comfy.model_management.get_torch_device()

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    real_model = None
    comfy.model_management.load_model_gpu(model)
    real_model = model.model

    noise = noise.to(device).requires_grad_()
    latent_image = latent_image.to(device).requires_grad_()

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    models = load_additional_models(positive, negative)

    sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    clip_encoded = clip.encode(clip_prompt)
    clip_target_embed = F.normalize(clip_encoded.float()).to(device)

    cfg_pred = comfy.samplers.CFGNoisePredictor(sampler.model)
    sampler.model_denoise = CLIPGuidedNoisePredictor(cfg_pred, vae, clip, clip_vision, clip_target_embed, clip_scale, device)
    if sampler.model.parameterization == "v":
        sampler.model_wrap = comfy.samplers.CompVisVDenoiser(sampler.model_denoise, quantize=True)
    else:
        sampler.model_wrap = k_diffusion_external.CompVisDenoiser(sampler.model_denoise, quantize=True)
    sampler.model_k = comfy.samplers.KSamplerX0Inpaint(sampler.model_wrap)
    sampler.sigma_min = float(sampler.model_wrap.sigma_min)
    sampler.sigma_max = float(sampler.model_wrap.sigma_max)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar)
    samples = samples.cpu()

    cleanup_additional_models(models)
    return samples
