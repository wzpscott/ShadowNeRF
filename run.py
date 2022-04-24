import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import pickle as pkl
import os, os.path as osp

# from load_blender import load_blender_data
from load_data import load_blender_data
from rays import Rays
from models import NeRF
from encoding import frequency_encode
from utils import *

# datadir = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/lego'
# half_res = True
# testskip = 8
# images, poses, render_poses, hwf, i_split = load_blender_data(
#     datadir, half_res=True, testskip=8)

exp_name = 'hotdog_relight_test'
base_dir = './logs'
log_dir = osp.join(base_dir, exp_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(osp.join(log_dir, 'val'), exist_ok=True)
os.makedirs(osp.join(log_dir, 'test'), exist_ok=True)

log_file_dir = osp.join(log_dir,'output.log')
fp = open(log_file_dir, 'w')
fp.write(f'output log for {exp_name}')

datadir = './data/nerf_synthetic_relight/hotdog'
images, cam_poses, light_poses, hwf, i_splits = load_blender_data(datadir, include_light=True, resize_factor=10)
images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:]) # white background

near = 2.
far = 6.
H, W, focal = hwf

# start training
NUM_EPOCHS = 10
BATCH_SIZE = 1024
NUM_SAMPLES = 128
# NUM_SAMPLES_UNIFORM = 64
# NUM_SAMPLES_IMPORTANCE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
L_x, L_dir = 10, 4
lrate = 5 * 1e-4
lrate_decay = 500
decay_rate = 0.1
decay_steps = lrate_decay * 1000


model = NeRF(x_dim=2*L_x*3+3, dir_dim=2*L_dir*3+3)
model = model.to(DEVICE)
data = []
for img, c2w in zip(images, cam_poses):
    data.append(Rays(img, c2w, W, H, focal, near, far, DEVICE))

train_data = [data[i] for i in i_splits[0]]
val_data = [data[i] for i in i_splits[1]]
test_data = [data[i] for i in i_splits[2]]

def encode(pts, dirs, L_x, L_dir):
    return frequency_encode(pts, L_x), frequency_encode(dirs, L_dir)
encode_fn = partial(encode, L_x=L_x, L_dir=L_dir)
decode_fn = model.forward

grad_vars = list(model.parameters())
optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

iter = 0
for epoch in range(NUM_EPOCHS):
    write('--------------------------------------------------------------', fp)
    write('--------------------------------------------------------------', fp)
    write(f'EPOCH {epoch}/{NUM_EPOCHS}', fp)
    np.random.shuffle(train_data)
    for rays in tqdm(train_data):
        losses = []

        if epoch == 0:
            crop = True
        else:
            crop = False
        for i, rays_batch in enumerate(rays.batchify(BATCH_SIZE, crop)):
            rays_batch.sample_along_rays(NUM_SAMPLES)
            rays_batch.query(encode_fn, decode_fn)
            rays_batch.volume_render()
            # rays_batch.peek()

            rgb = rays_batch.properties['rgb']
            y = rays_batch.rgb_gt
            img_loss = img2mse(rgb, y)
            loss = img_loss
            losses.append(loss.detach())
            psnr = mse2psnr(img_loss.cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter += 1
            ###   update learning rate   ###
            new_lrate = lrate * (decay_rate ** (iter / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        loss = sum(losses)/len(losses)
        psnr = mse2psnr(loss)
        write(f'Iter: {iter+1} Loss: {loss.item()} PSNR: {psnr.item()}', fp)

    write(f'start evaluation on val data after epoch{epoch+1}...', fp)

    losses = []
    for rays in tqdm(val_data):
        rgb_pred, depth_pred = eval_image(rays, encode_fn, decode_fn, BATCH_SIZE, NUM_SAMPLES)
        rgb_gt = rays.images.cpu()
        losses.append(img2mse(rgb_pred, rgb_gt))
        break
    loss = sum(losses)/len(losses)
    psnr = mse2psnr(loss)
    write(f'Epoch: {epoch+1} Loss: {loss.item()} PSNR: {psnr.item()}', fp)   

    rgb, depth, rgb_gt = rgb_pred.reshape(W, H, 3), depth_pred.reshape(W, H), rgb_gt.reshape(W, H, 3)
    plot([rgb, depth, rgb_gt, np.ones_like(rgb_gt)], 2, 2 , save_dir=f'{log_dir}/val/epoch_{epoch+1}.png')


write('--------------------------------------------------------------', fp)
write('--------------------------------------------------------------', fp)
write(f'start evaluation on test data...', fp)
losses = []
for i, rays in enumerate(tqdm(test_data)):
    rgb_pred, depth_pred = eval_image(rays, encode_fn, decode_fn, BATCH_SIZE, NUM_SAMPLES)
    rgb_gt = rays.images.cpu()
    rgb_pred, depth_pred, rgb_gt = rgb_pred.reshape(W, H, 3), depth_pred.reshape(W, H), rgb_gt.reshape(W, H, 3)
    plot([rgb_pred, depth_pred, rgb_gt, np.ones_like(rgb_gt)], 2, 2, save_dir=f'{log_dir}/test/{i}.png')
    losses.append(img2mse(rgb_pred, rgb_gt))
loss = sum(losses)/len(losses)
psnr = mse2psnr(loss)
write(f'Test: Loss: {loss.item()} PSNR: {psnr.item()}', fp)   





