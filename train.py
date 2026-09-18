import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from seisDataset.data_vision import data_vision
from seisDataset import seisDataloader, seistestdataloader
import matplotlib.pyplot as plt
import matplotlib
import openpyxl
matplotlib.use("Agg")

def sr(path):
# if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    # parser.add_argument('--data_dir', default="dataset\seistestsyn", type=str, help='dataset path')
    parser.add_argument('--data_dir', default=path, type=str, help='dataset path')
    parser.add_argument('--img_size', default=128, type=str)
    parser.add_argument('--batchsize', default=1, type=str)
    parser.add_argument('--per_epoch_step', default=20000, type=str)
    parser.add_argument('--epochs', default=20, type=str)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--val_freq', default=5000, type=str)
    parser.add_argument('--val_step', default=1, type=str)
    parser.add_argument('--print_freq', default=10, type=str)
    parser.add_argument('--save_checkpoint_freq', default=5000, type=str)


    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None


    # train_loader = torch.utils.data.DataLoader(seisDataloader.seis_dataset(args=args, train=True),
    #                                            batch_size=args.batchsize,
    #                                            num_workers=args.num_workers,
    #                                            pin_memory=True,
    #                                            shuffle=True)
    # val_loader = torch.utils.data.DataLoader(seisDataloader.seis_dataset(args=args, train=False),
    #                                          batch_size=args.batchsize,
    #                                          num_workers=args.num_workers,
    #                                          pin_memory=True,
    #                                          shuffle=True)
    
    seistest_loader = torch.utils.data.DataLoader(seistestdataloader.seis_dataset(args=args, train=False),
                                             batch_size=args.batchsize,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             shuffle=True)

    logger.info('Initial Dataset Finished')
    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        train_loader = torch.utils.data.DataLoader(seisDataloader.seis_dataset(args=args, train=True),
                                               batch_size=args.batchsize,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               shuffle=True)
        val_loader = torch.utils.data.DataLoader(seisDataloader.seis_dataset(args=args, train=False),
                                             batch_size=args.batchsize,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             shuffle=True)
        #while current_step < n_iter:
        for i in range(args.epochs):
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                # if current_step > n_iter:
                #     break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % args.print_freq == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                #if 1:
                if current_step % args.val_freq == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()

                        sr_img = Metrics.tensor2img3D(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img3D(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img3D(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img3D(visuals['INF'])  # uint8

                        # generation
                        plt.figure(figsize=(10,10))
                        plt.subplot(221)
                        plt.title('org')
                        plt.imshow(hr_img.astype(np.float32)/255.)

                        plt.subplot(222)
                        plt.title('out')
                        plt.imshow(sr_img.astype(np.float32)/255.)

                        plt.subplot(223)
                        plt.title('lr')
                        plt.imshow(lr_img.astype(np.float32)/255.)

                        plt.subplot(224)
                        plt.title('input')
                        plt.imshow(fake_img.astype(np.float32)/255.)
                        plt.savefig('{}/{}_{}.png'.format(result_path, current_step, idx))
                        plt.clf()

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % args.save_checkpoint_freq == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_fid = 0.0
        # corrupt image psnr and ssim
        l_avg_psnr = 0.0
        l_avg_ssim = 0.0
        l_avg_fid = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(seistest_loader):
            idx += 1
            print("the {}".format(idx))
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            # diffusion.ddim_test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img3D(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img3D(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img3D(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img3D(sr_img[iter]),
                        '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img3D(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img3D(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            # 3D data vision
            # data_vision(visuals['SR'], "SR")
            # data_vision(visuals['HR'], "HR")

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            
            sr_img = visuals['SR'][-1]
            sr_img_corrupt = visuals['SR'][0]
            # sr_img = visuals['SR'][-2:]
            # sr_img_corrupt = visuals['SR'][:2]
            hr_img = visuals['HR']

            np.save("{}/sr_filed_{}".format(os.path.dirname(path), os.path.basename(path)), sr_img)
            # eval         
            eval_psnr, eval_ssim, eval_fid, l_eval_psnr, l_eval_ssim, l_eval_fid = Metrics.calculate_all(sr_img, sr_img_corrupt, hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            avg_fid += eval_fid

            l_avg_psnr += l_eval_psnr
            l_avg_ssim += l_eval_ssim
            l_avg_fid += l_eval_fid

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, sr_img, hr_img, eval_psnr,
                                           eval_ssim, l_eval_psnr, l_eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_fid = avg_fid / idx

        l_avg_psnr = l_avg_psnr / idx
        l_avg_ssim = l_avg_ssim / idx
        l_avg_fid = l_avg_fid / idx

        # print("idx:{}".format(idx))

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation # FID: {:.4e}'.format(avg_fid))

        logger.info('# Corrupt data # PSNR: {:.4e}'.format(l_avg_psnr))
        logger.info('# Corrupt data # SSIM: {:.4e}'.format(l_avg_ssim))
        logger.info('# Corrupt data # FID: {:.4e}'.format(l_avg_fid))

        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}, fid: {:.4e}, corrupt_psnr: {:.4e}, corrupt_ssim: {:.4e}, corrupt_fid: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim, avg_fid, l_avg_psnr, l_avg_ssim, l_avg_fid))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim),
                'FID': float(avg_fid),
                'Corrupt_PSNR': float(l_avg_psnr),
                'Corrupt_SSIM': float(l_avg_ssim),
                'Corrupt_FID': float(l_avg_fid)
            })

# def find_npy_files(root_dir):
#     npy_files = []
    
#     for foldername, subfolders, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename.startswith('filed_piece') and filename.endswith('.npy'):
#                 npy_files.append(os.path.join(foldername, filename))
    
#     return npy_files

# root_directory = "C:/Users/20649/Seismic/seis_diffusion_SR/seis_diffusion_SR/dataset/big_pieces_recombined/filedata/test3"
# npy_file_paths = find_npy_files(root_directory)

# for path in npy_file_paths:
#     sr(path)
#     print(path)
path = "C:/Users/20649/Seismic/seis_diffusion_SR/seis_diffusion_SR/dataset/big_pieces_recombined/filedata/Original_F3_347.npy"
sr(path)
