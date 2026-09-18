import torch
from torchvision.transforms import functional as F
import numpy as np
import random
import torch.nn.functional as nnF
import cv2
import math

def downscale_img(seis, scale=(0.15, 0.75)):
    print("downscale")
    _, t, h, w = seis.shape
    modes = ['trilinear', 'nearest']
    scale1 = random.uniform(*scale)
    scale2 = random.uniform(*scale)
    scale3 = random.uniform(*scale)
    mode1 = random.sample(modes, 1)
    mode2 = random.sample(modes, 1)
    seis = nnF.interpolate(seis[None], scale_factor=(scale1, scale2, scale3), mode=mode1[0])
    seis = nnF.interpolate(seis, size=(t, h, w), mode=mode2[0])[0]
    return seis

def RotateBound(data, angle):
    scale = math.sin(abs(angle) * math.pi / 180) + math.cos(abs(angle) * math.pi / 180)
    (h, w) = data.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(data, M, (nW, nH))


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def z_score_clip(data, clp_s=3.2):
    z = (data - np.mean(data)) / np.std(data)
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))


def tensor_normalization(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range


def tensor_z_score_clip(data, clp_s=3.2):
    z = (data - torch.mean(data)) / torch.std(data)
    return tensor_normalization(torch.clip(z, min=-clp_s, max=clp_s))


def RandomGammaTransfer(seismic, p=0.5):
    if random.random() < p:  return seismic
    s_max, s_min = torch.max(seismic), torch.min(seismic)

    if random.randint(0, 1):
        gamma = random.uniform(0.6667, 1)
    else:
        gamma = random.uniform(1, 1.5)
    gamma_seismic = (seismic - s_min) ** gamma

    gamma_range = torch.max(gamma_seismic) - torch.min(gamma_seismic)
    gamma_seismic = ((gamma_seismic - torch.min(gamma_seismic)) / gamma_range) * (s_max - s_min) + s_min
    return gamma_seismic


def RandomResize(seis, tm=16, scale=(0.6667, 1.25), p=0.5):
    if random.random() < p: return seis
    B, C, T, H, W = seis.shape
    # ================================================
    if random.randint(0, 1):
        t_scale = random.uniform(scale[0], 1)
    else:
        t_scale = random.uniform(1, scale[1])
    resize_t = round(T * t_scale)
    if resize_t % tm != 0:
        resize_t = int(resize_t + tm - resize_t % tm)
    # ================================================
    if random.randint(0, 1):
        i_scale = random.uniform(scale[0], 1)
    else:
        i_scale = 1
    resize_i = round(H * i_scale)
    if resize_i % tm != 0:
        resize_i = int(resize_i + tm - resize_i % tm)
    # ================================================
    if random.randint(0, 1):
        x_scale = random.uniform(scale[0], 1)
    else:
        x_scale = 1
    resize_x = round(W * x_scale)
    if resize_x % tm != 0:
        resize_x = int(resize_x + tm - resize_x % tm)
    # ================================================
    resize_seis = nnF.interpolate(seis, (resize_t, resize_i, resize_x), mode='trilinear', align_corners=True)
    return resize_seis


def RandomHorizontalFlipCoord(*aug_list, p=0.5):
    if random.random() < p:
        return [F.hflip(data) for data in aug_list]
    return [data for data in aug_list]


def RandomVerticalFlipCoord(*aug_list, p=0.5):
    if random.random() < p:
        return [F.vflip(data) for data in aug_list]
    return [data for data in aug_list]


def RandomTimeflipCoord(*aug_list, p=0.5):
    if random.random() < p:
        return [F.vflip(data.permute((0, 2, 1, 3))).permute((0, 2, 1, 3)) for data in aug_list]
    return [data for data in aug_list]


def RandomRotateCoord(*aug_list, p=0.5):
    if random.random() < p:
        return [data.permute((0, 1, 3, 2)) for data in aug_list]
    return [data for data in aug_list]


def RandomRotateAgSynTline(seis, fault, p=0.25):
    if random.random() < p:
        return seis, fault
    _, cube_size, _, _ = seis.shape
    seis = seis[0].numpy()
    fault = fault[0].numpy()
    angle = random.randint(-45, 45)

    seis = RotateBound(seis.transpose((1, 2, 0)), angle).transpose((2, 0, 1))
    fault = RotateBound(fault.transpose((1, 2, 0)), angle).transpose((2, 0, 1))
    l = int((seis.shape[1] - cube_size) / 2)
    seis = seis[:, l:l + cube_size, l:l + cube_size]
    fault = (fault[:, l:l + cube_size, l:l + cube_size] > 0.2).astype(np.float32)
    return torch.from_numpy(seis)[None], torch.from_numpy(fault)[None]


def RandomRotateAgSynIXline(seis, fault, p=0.5):
    if random.random() < p:
        return seis, fault
    _, cube_size, _, _ = seis.shape
    seis = seis[0].numpy()
    fault = fault[0].numpy()
    angle = random.randint(-45, 45)
    seis = RotateBound(seis, angle)
    fault = RotateBound(fault, angle)
    l = int((seis.shape[1] - cube_size) / 2)
    seis = seis[l:l + cube_size, l:l + cube_size, :]
    fault = (fault[l:l + cube_size, l:l + cube_size, :] > 0.2).astype(np.float32)
    return torch.from_numpy(seis)[None], torch.from_numpy(fault)[None]


def RandomSynClip(seis, fault, syn_size=128, scale=1.5, p=0.75):
    if random.random() < p:
        return seis, fault
    t_scale = random.uniform(1, scale * 2.5)
    resize_t = round(syn_size * t_scale)

    i_scale = random.uniform(1, scale)
    resize_i = round(syn_size * i_scale)

    x_scale = random.uniform(1, scale)
    resize_x = round(syn_size * x_scale)

    resize_seis = nnF.interpolate(seis[None], (resize_t, resize_i, resize_x), mode='trilinear', align_corners=True)[0]
    resize_fault = (nnF.interpolate(fault[None], (resize_t, resize_i, resize_x), mode='nearest') > 0.2).float()[0]
    resize_seis, resize_fault = randomCrop(resize_seis, resize_fault)

    return resize_seis, resize_fault


def RandomSynShrink(syn_seis, syn_size=128, scale=0.5, p=0.5):
    if random.random() < p: return syn_seis
    t_scale = random.uniform(scale, 1)
    resize_t = round(syn_size * t_scale)
    if resize_t % 16 != 0:
        resize_t = int(resize_t + 16 - resize_t % 16)

    i_scale = random.uniform(1. - scale ** 2, 1)
    resize_i = round(syn_size * i_scale)
    if resize_i % 16 != 0:
        resize_i = int(resize_i + 16 - resize_i % 16)

    x_scale = random.uniform(1. - scale ** 2, 1)
    resize_x = round(syn_size * x_scale)
    if resize_x % 16 != 0:
        resize_x = int(resize_x + 16 - resize_x % 16)

    resize_syn_seis = nnF.interpolate(syn_seis, (resize_t, resize_i, resize_x), mode='trilinear', align_corners=True)
    return resize_syn_seis


def randomCrop(seis, fault, size=(128, 128, 128)):
    shape = seis[0].shape
    size = np.array(size)
    lim = shape - size
    w = random.randint(0, lim[0])
    h = random.randint(0, lim[1])
    c = random.randint(0, lim[2])
    return seis[:, w:w + size[0], h:h + size[1], c:c + size[2]], \
           fault[:, w:w + size[0], h:h + size[1], c:c + size[2]]


def RandomGaussianBlur_h(seismic, sigma_range, p=0.5):
    if random.random() < p:  return seismic
    sig_min, sig_max = sigma_range
    sigma_h = random.uniform(sig_min, sig_max)
    kernel_h = int(np.ceil(sigma_h) * 2 + 1)
    sigma_w = random.uniform(sig_min, sig_max)
    kernel_w = int(np.ceil(sigma_w) * 2 + 1)
    seismic = seismic.permute((0, 2, 1, 3))
    seismic = F.gaussian_blur(seismic, kernel_size=[kernel_h, kernel_w], sigma=[sigma_h, sigma_w])
    seismic = seismic.permute((0, 2, 1, 3))
    return seismic


def RandomGaussianBlur_w(seismic, sigma_range, p=0.5):
    if random.random() < p:  return seismic
    sig_min, sig_max = sigma_range
    sigma_h = random.uniform(sig_min, sig_max)
    kernel_h = int(np.ceil(sigma_h) * 2 + 1)
    sigma_w = random.uniform(sig_min, sig_max)
    kernel_w = int(np.ceil(sigma_w) * 2 + 1)
    seismic = seismic.permute((0, 3, 2, 1))
    seismic = F.gaussian_blur(seismic, kernel_size=[kernel_h, kernel_w], sigma=[sigma_h, sigma_w])
    seismic = seismic.permute((0, 3, 2, 1))
    return seismic


def RandomGaussianBlur_t(seismic, sigma_range, p=0.5):
    if random.random() < p:  return seismic
    sig_min, sig_max = sigma_range
    sigma_h = random.uniform(sig_min, sig_max)
    kernel_h = int(np.ceil(sigma_h) * 2 + 1)
    sigma_w = random.uniform(sig_min, sig_max)
    kernel_w = int(np.ceil(sigma_w) * 2 + 1)
    seismic = F.gaussian_blur(seismic, kernel_size=[kernel_h, kernel_w], sigma=[sigma_h, sigma_w])
    return seismic


def RandomGaussianBlur(seismic, p=0.25, sigma_range=[0.1, 2.5]):
    if random.random() < p:  return seismic
    aug_funcs = [RandomGaussianBlur_w, RandomGaussianBlur_h, RandomGaussianBlur_t]
    random.shuffle(aug_funcs)
    for func in aug_funcs:
        seismic = func(seismic, sigma_range)
    return seismic




def RandomGaussianNoise(seismic):
    print("RandomGaussianNoise")
    s_max, s_min = torch.max(seismic), torch.min(seismic)
    scale = random.uniform(0.2, 1.0) * (s_max - s_min) * 0.25
    noise = torch.normal(mean=0.0, std=scale, size=seismic.shape)
    seismic = seismic + noise
    seismic = torch.clip(seismic, min=s_min, max=s_max)
    return seismic


def RandomBlurNoise(seismic):
    print("RandomBlurNoise")
    s_max, s_min = torch.max(seismic), torch.min(seismic)
    scale = random.uniform(0.5, 1.0) * (s_max - s_min) * 1.8
    noise = torch.normal(mean=0.0, std=scale, size=seismic.shape)
    blur_list = [RandomGaussianBlur_w, RandomGaussianBlur_h]
    random.shuffle(blur_list)
    for func in blur_list:
        noise = func(noise, [1, 5], p=0)
    seismic = seismic + noise.to(seismic.device)
    seismic = torch.clip(seismic, min=s_min, max=s_max)
    return seismic


def RandomMorphologyNoise_h(noise, p=0.25):
    if random.random() < p: return noise
    noise = noise.permute((0, 3, 2, 1))
    kernel = cv2.getStructuringElement(random.randint(0, 2), (random.randint(1, 5), random.randint(1, 5)))
    if random.randint(0, 1):
        noise = cv2.dilate(noise.numpy()[0], kernel, iterations=1)
    else:
        noise = cv2.erode(noise.numpy()[0], kernel, iterations=1)
    noise = torch.from_numpy(noise)[None]
    noise = noise.permute((0, 3, 2, 1))
    return noise


def RandomMorphologyNoise_w(noise, p=0.25):
    if random.random() < p: return noise
    noise = noise.permute((0, 2, 1, 3))
    kernel = cv2.getStructuringElement(random.randint(0, 2), (random.randint(1, 5), random.randint(1, 5)))
    if random.randint(0, 1):
        noise = cv2.dilate(noise.numpy()[0], kernel, iterations=1)
    else:
        noise = cv2.erode(noise.numpy()[0], kernel, iterations=1)
    noise = torch.from_numpy(noise)[None]
    noise = noise.permute((0, 2, 1, 3))
    return noise


def RandomMorphologyNoise_t(noise, p=0.25):
    if random.random() < p: return noise
    kernel = cv2.getStructuringElement(random.randint(0, 2), (random.randint(1, 5), random.randint(1, 5)))
    if random.randint(0, 1):
        noise = cv2.dilate(noise.numpy()[0], kernel, iterations=1)
    else:
        noise = cv2.erode(noise.numpy()[0], kernel, iterations=1)
    noise = torch.from_numpy(noise)[None]
    return noise


def RandomMorphologyNoise(seismic):
    s_max, s_min = torch.max(seismic), torch.min(seismic)
    scale = random.uniform(0.5, 1.0) * (s_max - s_min) * 0.15
    noise = torch.normal(mean=0.0, std=scale, size=seismic.shape)
    morphology_list = [RandomMorphologyNoise_t, RandomMorphologyNoise_h, RandomMorphologyNoise_w]
    random.shuffle(morphology_list)
    for func in morphology_list:
        noise = func(noise)
    seismic = seismic + noise.to(seismic.device)
    seismic = torch.clip(seismic, min=s_min, max=s_max)
    return seismic


def RandomNoise(seismic, p=1):
    flag = random.randint(0, 2)
    if flag:
        seismic = RandomGaussianNoise(seismic)
    elif flag == 1:
        seismic = RandomMorphologyNoise(seismic)
    else:
        seismic = RandomBlurNoise(seismic)
    return seismic
