import torch
import tensorflow as tf
import os
import numpy as np
from torchvision.utils import make_grid, save_image
#import tensorflow_datasets as tfds


def merge_xy(x,y):
  return torch.cat([x[:,0:1,:,:], y[:,0:1,:,:],x[:,1:2,:,:],y[:,1:2,:,:],x[:,2:3,:,:], y[:,2:3,:,:]], 1)

def save_batch(sample,sample_dir,image_name):
  tf.io.gfile.makedirs(sample_dir)
  nrow = int(np.sqrt(sample.shape[0]))
  image_grid = make_grid(sample, nrow, padding=2)
  #sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
  #print(f"sample shape: {sample.shape}")
  with tf.io.gfile.GFile(os.path.join(sample_dir, image_name), "wb") as fout:
      save_image(image_grid, fout)

def save_separado(sample,sample_dir,image_name,batch_size, seq):
  tf.io.gfile.makedirs(sample_dir)
  nrow = 1
  for i in range(batch_size):
    image_grid = make_grid(sample[i,:,:,:], nrow, padding=2)
    with tf.io.gfile.GFile(os.path.join(sample_dir, f"{batch_size*seq+i}-"+image_name), "wb") as fout:
      save_image(image_grid, fout)

