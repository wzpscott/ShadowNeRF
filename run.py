import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import pickle as pkl

from load_blender import load_blender_data
from rays import Rays
from models import NeRF
from encoding import frequency_encode
from utils import *

datadir = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/lego'
half_res = True
testskip = 8
images, poses, render_poses, hwf, i_split = load_blender_data(
    datadir, half_res=True, testskip=8)
images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:]) # white background

near = 2.
far = 6.
H, W, focal = hwf

# start training
NUM_EPOCHS = 100
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
for img, c2w in zip(images, poses):
    data.append(Rays(img, c2w, W, H, focal, near, far, DEVICE))

train_data = [data[i] for i in i_split[0]]
val_data = [data[i] for i in i_split[1]]
test_data = [data[i] for i in i_split[2]]

def encode(pts, dirs, L_x, L_dir):
    return frequency_encode(pts, L_x), frequency_encode(dirs, L_dir)
encode_fn = partial(encode, L_x=L_x, L_dir=L_dir)
decode_fn = model.forward

grad_vars = list(model.parameters())
optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

iter = 0
for epoch in range(NUM_EPOCHS):
    tqdm.write('--------------------------------------------------------------')
    tqdm.write('--------------------------------------------------------------')
    tqdm.write(f'EPOCH {epoch}/{NUM_EPOCHS}')
    np.random.shuffle(train_data)
    for rays in tqdm(train_data):
        losses = []
        for i, rays_batch in enumerate(rays.batchify(BATCH_SIZE)):
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
        tqdm.write(f'Iter: {iter+1} Loss: {loss.item()} PSNR: {psnr.item()}')

    tqdm.write(f'start evaluation on val data after epoch{epoch+1}...')
    losses = []
    for rays in tqdm(val_data):
        rgb_pred, depth_pred = eval_image(rays, encode_fn, decode_fn, BATCH_SIZE, NUM_SAMPLES)
        rgb_gt = rays.images.cpu()
        losses.append(img2mse(rgb_pred, rgb_gt))
        break
    loss = sum(losses)/len(losses)
    psnr = mse2psnr(loss)
    tqdm.write(f'Epoch: {epoch+1} Loss: {loss.item()} PSNR: {psnr.item()}')   

    rgb, depth = rgb_pred.reshape(W, H, 3), depth_pred.reshape(W, H)
    plot([rgb, depth], 1, 2, save_dir=f'results/epoch_{epoch+1}.png')


tqdm.write('--------------------------------------------------------------')
tqdm.write('--------------------------------------------------------------')
tqdm.write(f'start evaluation on test data...')
losses = []
for rays in tqdm(test_data):
    rgb_pred, depth_pred = eval_image(rays, encode_fn, decode_fn, BATCH_SIZE, NUM_SAMPLES)
    rgb_gt = rays.images.cpu()
    losses.append(img2mse(rgb_pred, rgb_gt))
loss = sum(losses)/len(losses)
psnr = mse2psnr(loss)
tqdm.write(f'Test: Loss: {loss.item()} PSNR: {psnr.item()}')   

rgb_pred, depth_pred, rgb_gt = rgb_pred.reshape(W, H, 3), depth_pred.reshape(W, H), rgb_gt.reshape(W, H, 3)
plot([rgb_pred, depth_pred, rgb_gt, np.ones_like(rgb_gt)], 2, 2, save_dir=f'results/Test.png')



