import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
import openpyxl


def tensor2img3D(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor + 1) / 2
    n_dim = tensor.dim()
    # print(n_dim)
    if n_dim == 4:
        n_img = len(tensor)
        tensor = tensor[:, :, 32, :]
        tensor = torch.from_numpy(plt.get_cmap('gray')(tensor.numpy())[:, :, :, :-1])
        img_np = make_grid(tensor.permute((0, 3, 1, 2)), nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        tensor = tensor[:, 32, :]
        img_np = plt.get_cmap('gray')(tensor.numpy())[:, :, :-1]
        #img_np = np.transpose(img_np, (2, 0, 1))  # HWC, RGB
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round() 
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    # rgb_img = img[:, :, :3]
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# def ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter3D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()


# def calculate_ssim(img1, img2):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')


def calculate_ssim(img1, img2):
    ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min())
    return ssim_value
    

def fid(img1, img2):

    # calculate mean and covariance statistics
    mu1, sigma1 = img1.mean(axis=0), np.cov(img1, rowvar=False)
    mu2, sigma2 = img2.mean(axis=0), np.cov(img2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    # if iscomplexobj(covmean):
    covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# calculate frechet inception distance
def calculate_fid(img1, img2):

    # Reshape the 3D images to 2D arrays (flatten along the third dimension)
    img1_flat = img1.reshape(-1, img1.shape[-1])
    img2_flat = img2.reshape(-1, img2.shape[-1])

    fid_value = fid(img1_flat, img2_flat)
    return fid_value
    

def calculate_all(img1, img2, img3, out_type=np.uint8, min_max=(-1, 1)):

    img1 = img1.squeeze().float().cpu().clamp_(*min_max)  # clamp
    img2 = img2.squeeze().float().cpu().clamp_(*min_max)  # clamp
    img3 = img3.squeeze().float().cpu().clamp_(*min_max)  # clamp

    if out_type == np.uint8:
        imgnorm1 = (((img1 + 1) / 2) * 255.0).round()
        imgnorm2 = (((img2 + 1) / 2) * 255.0).round()
        imgnorm3 = (((img3 + 1) / 2) * 255.0).round()

    workbook = openpyxl.load_workbook("test_result.xlsx")
    sheet = workbook['Sheet1']
    last_row = sheet.max_row + 1

    # np.save("sr_img_{}.npy".format(last_row), img1)
    # np.save("sr_img_corrupt_{}.npy".format(last_row), img2)
    # np.save("hr_img_{}.npy".format(last_row), img3)

    # psnr,ssim [0,255]
    # sr_img
    img_psnr = calculate_psnr(imgnorm1.numpy().astype(out_type), imgnorm3.numpy().astype(out_type))
    img_ssim = calculate_ssim(imgnorm1.numpy().astype(out_type), imgnorm3.numpy().astype(out_type))
    img_fid = calculate_fid(img1.numpy(), img3.numpy())

    # sr_img_corrupt
    l_img_psnr = calculate_psnr(imgnorm2.numpy().astype(out_type), imgnorm3.numpy().astype(out_type))
    l_img_ssim = calculate_ssim(imgnorm2.numpy().astype(out_type), imgnorm3.numpy().astype(out_type))
    l_img_fid = calculate_fid(img2.numpy(), img3.numpy())

    sheet[f'A{last_row}'] = img_psnr
    sheet[f'B{last_row}'] = img_ssim
    sheet[f'C{last_row}'] = img_fid
    sheet[f'D{last_row}'] = l_img_psnr
    sheet[f'E{last_row}'] = l_img_ssim
    sheet[f'F{last_row}'] = l_img_fid

    workbook.save("test_result.xlsx")

    return img_psnr, img_ssim, img_fid, l_img_psnr, l_img_ssim, l_img_fid


def diffusion_all(img1, img2, index, out_type=np.uint8, min_max=(-1, 1)):

    img1 = img1.squeeze().float().cpu().clamp_(*min_max)  # clamp
    img2 = img2.squeeze().float().cpu().clamp_(*min_max)  # clamp

    if out_type == np.uint8:
        imgnorm1 = (((img1 + 1) / 2) * 255.0).round()
        imgnorm2 = (((img2 + 1) / 2) * 255.0).round()

    workbook = openpyxl.load_workbook("sample_fid.xlsx")
    sheet = workbook['Sheet1']
    last_row = sheet.max_row + 1
    
    s_psnr = calculate_psnr(imgnorm1.numpy().astype(out_type), imgnorm2.numpy().astype(out_type))
    s_ssim = calculate_ssim(imgnorm1.numpy().astype(out_type), imgnorm2.numpy().astype(out_type))
    s_fid = calculate_fid(img1.numpy(), img2.numpy())

    sheet[f'A{last_row}'] = s_psnr
    sheet[f'B{last_row}'] = s_ssim
    sheet[f'C{last_row}'] = s_fid
    sheet[f'D{last_row}'] = index

    workbook.save("sample_fid.xlsx")

    return s_psnr, s_ssim, s_fid