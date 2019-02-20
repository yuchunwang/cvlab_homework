from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf

import util
from model import CCPDRegressor


class CCPDTest:
    def __init__(self, img_dir, img_size):
        img_paths = Path(img_dir).glob('*.jpg')
        self.img_paths = sorted(list(img_paths))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(self.img_size)
        img = tf.to_tensor(img)
        return img


test_set = CCPDTest('./ccpd5000/test/', (192, 320))
test_loader = DataLoader(test_set, 32, shuffle=False, num_workers=2)

device = 'cpu'
model = CCPDRegressor().to(device)
model.load_state_dict(torch.load('log/2019.02.19-09:11:11/model.pth'))
model.eval()

log_dir = Path('./test/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)
print(log_dir)


def test(pbar):
    anns = []
    for img_b in iter(test_loader):
        kpt_b = model(img_b.to(device)).cpu()

        for img, kpt in zip(img_b, kpt_b):
            img = tf.to_pil_image(img)
            vis = util.draw_plate(img, kpt)
            vis = util.draw_kpts(vis, kpt, c='red')
            vis.save(log_dir / f'{pbar.n:03d}_vis.jpg')

            anns.append([f'{pbar.n:03d}.jpg', *kpt.numpy().tolist()])
            pbar.update()

    return pd.DataFrame(anns)


with torch.no_grad():
    with tqdm(total=len(test_set)) as pbar:
        df_pred = test(pbar)
    df_pred.columns = ['name', 'BR_x', 'BR_y', 'BL_x', 'BL_y', 'TL_x', 'TL_y', 'TR_x', 'TR_y']
    df_pred.to_csv(log_dir / 'test_pred.csv', float_format='%.5f', index=False)

