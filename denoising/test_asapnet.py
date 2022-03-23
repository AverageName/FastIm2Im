import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict
# from scipy.io import loadmat

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN

@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo          # model_zoo
   |--dncnn_15        # model_name
   |--dncnn_25
   |--dncnn_50
   |--dncnn_gray_blind
   |--dncnn_color_blind
   |--dncnn3
|--testset            # testsets
   |--set12           # testset_name
   |--bsd68
   |--cbsd68
|--results            # results
   |--set12_dncnn_15  # result_name = testset_name + '_' + model_name
   |--set12_dncnn_25
   |--bsd68_dncnn_15
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='asap', help='')
    #parser.add_argument('--model_path', type=str, default='checkpoints/asap.pth', help='')
    parser.add_argument('--model_path', type=str, default='denoising/asap/models/3200_G.pth', help='')
    parser.add_argument('--target_path', type=str, default='testsets/bsdH', help='')
    parser.add_argument('--noisy_path', type=str, default='testsets/bsdL', help='')
    parser.add_argument('--result_path', type=str, default='testsets/bsdL_predict', help='')
    args = parser.parse_args()

    n_channels = 3 
    nb = 17 

    result_name = args.model_name     # fixed
    border = 0        # shave boader to calculate PSNR and SSIM
    model_path = args.model_path #'denoising/archive_models/dncnn_val50/models/5600_G.pth'

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = args.noisy_path  # 'testsets/val_15' # L_path, for Low-quality images
    H_path = args.target_path  #'testsets/bsdH'  # H_path, for High-quality images
    E_path = args.result_path  # 'testsets/val_25_predict'   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------


    from models.network_asapnet import ASAPNetsGenerator as net
    from options.train_asap import TrainOptions
    opt2 = TrainOptions().parse()
    model = net(opt2, in_nc=n_channels, out_nc=n_channels)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path)# if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        img_L = util.single2tensor3(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        img_L = util.preprocess_img(img_L)
        
        img_E = model(img_L)

        img_E = util.tensor2uint(img_E)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()
            img_H = util.preprocess_img(img_H)
        
            # --------------------------------
            # PSNR and SSIM
            # --------------------------------
            print(img_E[:, :64].shape, img_H.shape)
            psnr = util.calculate_psnr(img_E[:, :64], img_H, border=border)
            ssim = util.calculate_ssim(img_E[:, :64], img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
