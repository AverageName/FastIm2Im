import numpy as np
import random
import shutil
from skimage import io
import os
import cv2

seed=42
random.seed(seed)

#randomly split data into train and validation parts (train - 432 images,val - 68 images)
def bsd_preprocessing():
    n_to_train=432
    path='./BSD500'
    train_path='./train'
    val_path='./val'
    
    all_ims = os.listdir(path)
    ims_to_train = random.sample(all_ims, n_to_train)
    
    if not os.path.exists(train_path):
            os.makedirs(train_path)

    for i in range(len(ims_to_train)):
        dst = os.path.join(train_path, ims_to_train[i])
        src = os.path.join(path, ims_to_train[i])
        shutil.move(src, dst)
    
    ims_to_val = os.listdir(path)
    
    if not os.path.exists(val_path):
            os.makedirs(val_path)

    for i in range(len(ims_to_val)):
        dst = os.path.join(val_path, ims_to_val[i])
        src = os.path.join(path, ims_to_val[i])
        shutil.move(src, dst)

def create_noisy_ims(path,sigma):
    all_ims = os.listdir(path)
    dir_name='./'+path[path.rfind('/')+1:]+'_'+str(sigma)
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    for image_name in all_ims:
        image = io.imread(os.path.join(path,image_name))
        
        gauss_noise = np.random.normal(loc=0,scale=sigma,size=image.shape)
        noisy_image = image + gauss_noise
        out_image=np.zeros(image.shape)
        cv2.normalize(noisy_image, out_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        out_image = out_image.astype(np.uint8)
        
        io.imsave(os.path.join(dir_name, image_name),out_image) 



