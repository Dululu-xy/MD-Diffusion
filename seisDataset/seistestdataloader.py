from torch.utils.data import Dataset
import os
import numpy as np
import torch
import random
import torch.nn.functional as F
from seisDataset.data_aug import RandomGammaTransfer, tensor_z_score_clip, RandomHorizontalFlipCoord, \
    RandomVerticalFlipCoord, \
    RandomRotateCoord, RandomRotateAgSynTline, RandomGaussianBlur, RandomNoise, z_score_clip, RandomRotateAgSynIXline, downscale_img


class seis_dataset(Dataset):
    def __init__(self, args, train=True):
        self.train = train
        self.path = args.data_dir
        self.args = args
        # self.data_list = [os.path.join(args.data_dir, fpath) for fpath in os.listdir(args.data_dir)]
        self.target_size = (args.img_size,args.img_size,args.img_size)
        self.crop_range = (int(args.img_size / 1.5), int(args.img_size * 2))
        self.pos_aug_list = [
            RandomRotateCoord,
            RandomVerticalFlipCoord,
            RandomHorizontalFlipCoord,
            RandomRotateAgSynTline,
            RandomRotateAgSynIXline,
        ]
        self.vox_aug_list = [
            downscale_img,
            RandomNoise,
            # RandomGammaTransfer,
            # RandomGaussianBlur
        ]

    def __len__(self):
        if self.train:
            return self.args.per_epoch_step * self.args.batchsize
        else:
            return self.args.val_step * self.args.batchsize

    def __getitem__(self, index):

        print(self.path)
        full_seis = np.load(self.path).astype(np.float32)
        
        # sample_t = random.randint(self.crop_range[0], self.crop_range[1])
        # sample_h = random.randint(self.crop_range[0], self.crop_range[1])
        # sample_w = random.randint(self.crop_range[0], self.crop_range[1])

        # T, H, W = full_seis.shape

        # start_t = random.randint(0, np.clip(T - sample_t, a_min=0, a_max=999))
        # start_h = random.randint(0, np.clip(H - sample_h, a_min=0, a_max=999))
        # start_w = random.randint(0, np.clip(W - sample_w, a_min=0, a_max=999))

        # clear_seis = full_seis[start_t:start_t + sample_t, start_h:start_h + sample_h, start_w:start_w + sample_w]

        # clear_seis = torch.from_numpy(z_score_clip(clear_seis.astype(np.float32)))[None]
        # clear_seis  = F.interpolate(clear_seis[None], size=self.target_size, mode='trilinear', align_corners=True)[0]

        # random.shuffle(self.pos_aug_list)
        # random.shuffle(self.vox_aug_list)
        # for aug_func in self.pos_aug_list:
        #     clear_seis, _ = aug_func(clear_seis, torch.zeros_like(clear_seis))
        # for aug_func in self.vox_aug_list:
        #     noise_seis = aug_func(clear_seis)

        # clear_seis = torch.clip(clear_seis * 2 - 1, min=-1, max=1)  # let range == -1,1
        # noise_seis = torch.clip(noise_seis * 2 - 1, min=-1, max=1)
        # clear_seis = torch.from_numpy(full_seis.astype(np.float32)).unsqueeze(dim=0)
        # filed data z_score_clip(full_seis.astype(np.float32))
        clear_seis = torch.from_numpy(z_score_clip(full_seis.astype(np.float32))).unsqueeze(dim=0)
        print(clear_seis.shape)

        return {'HR': clear_seis, 'SR': clear_seis, 'Index': index}
