"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import glob
import random


class BSDDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=3)
        parser.set_defaults(aspect_ratio=1.0)
        parser.set_defaults(batchSize=16)
        parser.set_defaults(lr_instance=True)

        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')

        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(preprocess_mode='scale_width_and_crop')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        #label_dir = os.path.join(root, 'gtFine', phase)
        label_dir = os.path.join(root, '%s_labels'%phase)
        label_paths = sorted(glob.glob(os.path.join(label_dir, '**', '*.jpg'), recursive=True))

        image_dir = os.path.join(root, '%s_images'%phase)
        image_paths = sorted(glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True))

        instance_paths = []

        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return name1.split('.')[-1] == name2.split('.')[-1]
