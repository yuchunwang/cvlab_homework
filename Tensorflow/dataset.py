from PIL import Image
from tqdm import tqdm
import os
import tensorflow as tf
import numpy as np

sess=tf.Session()

class CCPD5000_data:
  def __init__(self, img_dir):
    self.img_names= os.listdir( img_dir )
    self.img_names = sorted(self.img_names)
    self.img_dir = img_dir
    
  def __len__(self):
    return len(self.img_names)
  

  def shape(self, idx):
    img = __getitem__(idx)
    return img.shape
    
  def __getitem__(self, idx):
    img_name = self.img_names[idx]
    img_path = os.path.join(self.img_dir,img_name)
    img = Image.open(img_path)
    W, H = img.size
    img = img.convert('RGB')
    img = img.resize((192, 320))
    img=np.array(img)
#     img = tf.convert_to_tensor(img)
#     img = tf.cast(img, tf.float32)
    return img

class CCPD5000_labels:
  def __init__(self, img_dir):
    self.img_names= os.listdir( img_dir )
    self.img_names = sorted(self.img_names)
    self.img_dir = img_dir
    
  def __len__(self):
    return len(self.img_names)

  def shape(self, idx):
    kpt = __getitem__(idx)
    return kpt.shape

  def __getitem__(self, idx):
    img_name = self.img_names[idx]
    img_path = os.path.join(self.img_dir,img_name)
    img = Image.open(img_path)
    W, H = img.size
    img = img.convert('RGB')
    img = img.resize((192, 320))

    token = img_name.split('-')[3]
    token = token.replace('&', '_')
    kpt = [float(val) for val in token.split('_')] # [8,]
    kpt = np.reshape(kpt,[4, 2]) # [4, 2]
    kpt = kpt / np.array([W, H])
    kpt = np.reshape(kpt,[-1])# [8,] 
#     kpt = tf.cast(kpt,tf.float32)
    return kpt



