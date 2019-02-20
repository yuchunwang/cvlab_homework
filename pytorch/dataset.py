from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf

class CCPD5000:
  def __init__(self, img_dir):
    self.img_dir = Path(img_dir)
    self.img_paths = self.img_dir.glob('*.jpg')
    self.img_paths = sorted(list(self.img_paths))
    
  def __len__(self):
    return len(self.img_paths)
  
  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    
    # load image
    img = Image.open(img_path)
    W, H = img.size
    img = img.convert('RGB')
    img = img.resize((192, 320))
    img = tf.to_tensor(img)
    
    # parse annotation
    name = img_path.name
    token = name.split('-')[3]
    token = token.replace('&', '_')
    kpt = [float(val) for val in token.split('_')]
    kpt = torch.tensor(kpt) # [8,]
    kpt = kpt.view(4, 2) # [4, 2]
    kpt = kpt / torch.FloatTensor([W, H])
    kpt = kpt.view(-1) # [8,]
    
    return img, kpt
  

train_set = CCPD5000('./ccpd5000/train')
print(len(train_set))

img, kpt = train_set[-1]
print(img.size())
print(kpt.size())

