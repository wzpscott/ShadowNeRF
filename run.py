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
from rays import *
from models import NeRF, NePhong
from encoding import frequency_encode
from utils import *

# datadir = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/lego'
# half_res = True
# testskip = 8
# images, poses, render_poses, hwf, i_split = load_blender_data(
#     datadir, half_res=True, testskip=8)

exp_name = 'hotdog_1'
base_dir = './logs'
log_dir = osp.join(base_dir, exp_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(osp.join(log_dir, 'val'), exist_ok=True)
os.makedirs(osp.join(log_dir, 'test'), exist_ok=True)

log_file_dir = osp.join(log_dir,'output.log')
with open(log_file_dir, 'w') as fp:
    fp.write(f'output log for {exp_name}')

datadir = './data/nerf_synthetic_relight/hotdog'
images, cam_poses, light_poses, i_light_poses, hwf, i_splits = load_blender_data(
                                                                datadir, include_light=True, resize_factor=20)
images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:]) # white background

near = 2.
far = 6.
H, W, focal = hwf

# start training
NUM_EPOCHS = 100
cur_epoch = 0 # current epochs(for resuming training)
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

# model = NeRF(x_dim=2*L_x*3+3, dir_dim=2*L_dir*3+3)
model = NePhong(x_dim=2*L_x*3+3, dir_dim=2*L_dir*3+3)
grad_vars = list(model.parameters())
optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))
if osp.exists(osp.join(log_dir, 'checkpoint.pt')):
    checkpoint = torch.load(osp.join(log_dir, 'checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    cur_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
model = model.to(DEVICE)

data = []
for img, c2w, c2w_light, i_light in zip(images, cam_poses, light_poses, i_light_poses):
    data.append(CamRays(img, c2w, c2w_light, i_light, W, H, focal, near, far,  DEVICE))

train_data = [data[i] for i in i_splits[0]]
val_data = [data[i] for i in i_splits[1]]
test_data = [data[i] for i in i_splits[2]]
val_data = val_data + test_data

light_data = []
for i in range(10):
    light_data.append(LightRays(light_poses[i], W, H, focal, near, far, DEVICE))

iter = 0
render_list = ['normal', 'ambient', 'diffuse', 'specular']
for epoch in range(cur_epoch, NUM_EPOCHS):
    write('--------------------------------------------------------------', log_file_dir)
    write('--------------------------------------------------------------', log_file_dir)
    write(f'EPOCH {epoch+1}/{NUM_EPOCHS}', log_file_dir)
    write(f'start update depth maps', log_file_dir)

    with torch.no_grad():
        for rays in tqdm(light_data):
            rays.update_depth_map(BATCH_SIZE, NUM_SAMPLES, L_x, L_dir, model)

    write(f'start training', log_file_dir)    
    np.random.shuffle(train_data)
    for rays in tqdm(train_data):
        losses = []
        light_rays = light_data[rays.i_light]

        crop = True if epoch < 5 else False
        pretrain = True if epoch < 10 else False

        for i, rays_batch in enumerate(rays.batchify(BATCH_SIZE, random=True, crop=crop)):
            rays_batch.sample(NUM_SAMPLES)
            rays_batch.encode(L_x, L_dir)
            rays_batch.decode(model, pretrain=pretrain)
            rays_batch.volume_render(render_list)
            # rays_batch.peek()

            rgb = rays_batch.pixel_properties['rgb']
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
        write(f'Iter: {iter+1} Loss: {loss.item()} PSNR: {psnr.item()}', log_file_dir)

    write(f'start evaluation on val data after epoch{epoch+1}...', log_file_dir)

    # losses = []
    # for rays in tqdm(np.random.choice(val_data, 10, replace=False)):
    #     light_rays = light_data[rays.i_light]
    #     rgb_pred, depth_pred, normal_pred, ambient_pred, diffuse_pred, specular_pred, shadow_pred = eval_image(
    #         rays, L_x, L_dir, model, BATCH_SIZE, NUM_SAMPLES, light_rays)
    #     rgb_gt = rays.img_gt.cpu()
    #     losses.append(img2mse(rgb_pred, rgb_gt))

    # loss = sum(losses)/len(losses)
    # psnr = mse2psnr(loss)
    # write(f'Epoch: {epoch+1} Loss: {loss.item()} PSNR: {psnr.item()} shadow:{shadow_pred.mean().item()}', log_file_dir)   

    # rgb, depth, rgb_gt = rgb_pred.reshape(W, H, 3), depth_pred.reshape(W, H), rgb_gt.reshape(W, H, 3)
    # normal_pred, ambient_pred, diffuse_pred, specular_pred = normal_pred.reshape(W, H, 3), ambient_pred.reshape(W,H,3), diffuse_pred.reshape(W,H,3), specular_pred.reshape(W,H,3)
    
    # shadow_pred = shadow_pred.reshape(W, H)

    # plot([rgb, depth, rgb_gt, normal_pred, ambient_pred, diffuse_pred, specular_pred, shadow_pred], 
    #     2, 4 , save_dir=f'{log_dir}/val/epoch_{epoch+1}.png')

    losses = []
    for rays in tqdm(np.random.choice(val_data, 10, replace=False)):
        light_rays = light_data[rays.i_light]
        depth_cam, depth_light, rgb_cam, rgb_light, shadow, xs, x_gts = eval_shadow(
                                                    rays, light_rays, L_x, L_dir, model, BATCH_SIZE, NUM_SAMPLES)
        rgb_gt = rays.img_gt.cpu()
        losses.append(img2mse(rgb_cam, rgb_gt))

    loss = sum(losses)/len(losses)
    psnr = mse2psnr(loss)
    write(f'Epoch: {epoch+1} Loss: {loss.item()} PSNR: {psnr.item()}', log_file_dir)   

    xs, x_gts = xs[:, 2].reshape(W, H), x_gts[:W*H, 2].reshape(W, H)
    xs, x_gts = (xs-xs.min())/(xs.max()-xs.min()), (x_gts-x_gts.min())/(x_gts.max()-x_gts.min())
    depth_cam, depth_light,shadow = depth_cam.reshape(W, H), depth_light.reshape(W, H), shadow.reshape(W, H)
    rgb_cam, rgb_light, rgb_gt = rgb_cam.reshape(W, H, 3), rgb_light.reshape(W, H, 3), rgb_gt.reshape(W, H, 3)
    plot([rgb_cam, rgb_light, rgb_gt, depth_cam, depth_light, shadow, xs, x_gts], 
        2, 4 , save_dir=f'{log_dir}/val/epoch_{epoch+1}.png')


    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'{log_dir}/checkpoint.pt')

write('--------------------------------------------------------------', log_file_dir)
write('--------------------------------------------------------------', log_file_dir)
write(f'start final evaluation ...', log_file_dir)
losses = []
for i, rays in enumerate(tqdm(val_data)):
    rgb_pred, depth_pred, normal_pred, ambient_pred, diffuse_pred, specular_pred = eval_image(rays, L_x, L_dir, model, BATCH_SIZE, NUM_SAMPLES)
    rgb_gt = rays.img_gt.cpu()
    rgb_pred, depth_pred, rgb_gt = rgb_pred.reshape(W, H, 3), depth_pred.reshape(W, H), rgb_gt.reshape(W, H, 3)
    normal_pred, ambient_pred, diffuse_pred, specular_pred = \
        normal_pred.reshape(W, H, 3), ambient_pred.reshape(W,H,3), diffuse_pred.reshape(W,H,3), specular_pred.reshape(W,H,3)
    plot([rgb_pred, depth_pred, rgb_gt, normal_pred, ambient_pred, diffuse_pred, specular_pred, np.ones_like(rgb_gt)], 
        2, 4, save_dir=f'{log_dir}/test/{i}.png')
    losses.append(img2mse(rgb_pred, rgb_gt))
loss = sum(losses)/len(losses)
psnr = mse2psnr(loss)
write(f'Test: Loss: {loss.item()} PSNR: {psnr.item()}', log_file_dir)   





