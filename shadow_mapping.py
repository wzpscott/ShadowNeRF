import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from run_nerf_helpers import *
from run_nerf import *
import cv2
import pickle as pkl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = config_parser()
args = parser.parse_args()
def load_example_blender_data(basedir, n=2, half_res=True, testskip=1):
    splits = ['train']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    imgs = []
    poses = []  
    meta = metas[s]
    rand_idx = np.random.randint(0,100,size=(n))
    for i, frame in enumerate(meta['frames']):
        if i not in rand_idx:
            continue
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        img = (np.array(imageio.imread(fname)) / 255.).astype(np.float32)
        pose = np.array(frame['transform_matrix']).astype(np.float32)
        imgs.append(img)
        poses.append(pose)
    
    imgs = np.stack(imgs, 0)
    poses = np.stack(poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    res_factor = 4
    if half_res:
        H = H//res_factor
        W = W//res_factor
        focal = focal/res_factor

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    return imgs, poses, [H, W, focal]

def plot(imgs, m, n, save_dir=None):
    for i, img in enumerate(imgs):
        plt.subplot(m, n, i+1)
        plt.imshow(img)
        plt.axis('off')
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

images, poses, hwf, = load_example_blender_data(
            '/home/wzpscott/NeuralRendering/nerf-pytorch/data/nerf_synthetic/lego', n=10, half_res=True)
near = 2.
far = 6.
H, W, focal = hwf
H, W = int(H), int(W)
hwf = [H, W, focal]
print(f'Loaded images, images shape: {images.shape}, poses_shape: {poses.shape}, hwf: {hwf}')

render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

bds_dict = {
    'near': near,
    'far': far,
    }
K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

render_kwargs_train.update(bds_dict)
render_kwargs_test.update(bds_dict)

images, poses, K = torch.Tensor(images).to(device), torch.Tensor(poses).to(device), torch.Tensor(K).to(device)

render_poses = poses

if os.path.exists('test/rgbs.pkl') and os.path.exists('test/depths.pkl'):
    with open('test/rgbs.pkl','rb') as f:
        rgbs = pkl.load(f)
    with open('test/depths.pkl','rb') as f:
        depths = pkl.load(f)
else:
    with torch.no_grad():
        rgbs, _ , depths = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, render_factor=1)
        with open('test/rgbs.pkl','wb') as f:
            pkl.dump(rgbs, f)
        with open('test/depths.pkl','wb') as f:
            pkl.dump(depths, f)

cam_idx, light_idx = 3,1
c2w_cam, c2w_light = poses[cam_idx], poses[light_idx]
rgb_cam, rgb_light = rgbs[cam_idx], rgbs[light_idx]
depth_cam, depth_light = torch.Tensor(depths[cam_idx]), torch.Tensor(depths[light_idx])

def get_pts(c2w, depth):
    rays_o, rays_d = get_rays(H, W, K, c2w.cuda())
    rays_o, rays_d = rays_o.cpu(), rays_d.cpu()
    rays_d = F.normalize(rays_d, dim=2)

    depth = depth.cpu()
    pts = rays_o + depth[..., None] * rays_d
    pts = torch.cat([pts, torch.ones(H, W, 1)], dim=-1)
    return pts.reshape(-1,4)

def get_proj(f, near, far ,W, H):
    proj = torch.Tensor([
        [2*f/W, 0, 0, 0],
        [0, 2*f/H, 0, 0],
        [0, 0, -far/(far-near), -2*far*near/(far-near)],
        [0, 0, -1, 0]
    ])
    return proj

def plot_depth(pts, save_dir=''):
    pts = pts.reshape(-1, 4).detach().cpu().numpy()
    x,y,z = pts[:, 0], pts[:, 1], pts[:, 2]
    normalizer = pts[:, 3]

    x,y,z = x/normalizer, y/normalizer, z/normalizer
    plt.scatter(x, y, c=z)
    plt.savefig(save_dir)
    plt.close()

proj = get_proj(focal, near, far, W, H).cpu()

pts_cam = get_pts(c2w_cam, depth_cam)
pts_light = get_pts(c2w_light, depth_light)

depth_cam, depth_light = depth_cam.cpu(), depth_light.cpu()
c2w_cam, c2w_light = c2w_cam.cpu(), c2w_light.cpu()

w2c_cam = torch.linalg.inv(c2w_cam)
w2c_light = torch.linalg.inv(c2w_light)


def norm(x):
    N = x[...,3][...,None]
    return x/N

x = pts_cam @ w2c_light.T
x[:, 2] = x[:, 2].clip(-6,-2)
x = x @ proj.T
x = norm(x)

x_gt = pts_light @ w2c_light.T
x_gt[:, 2] = x_gt[:, 2].clip(-6,-2)
x_gt = x_gt @ proj.T
x_gt = norm(x_gt)

x_gt_cam = pts_cam @ w2c_cam.T
x_gt_cam[:, 2] = x_gt_cam[:, 2].clip(-6,-2)
x_gt_cam = x_gt_cam @ proj.T
x_gt_cam = norm(x_gt_cam)

plot_depth(x, save_dir='test/x.png')
plot_depth(x_gt_cam, save_dir='test/x_gt_cam.png')
plot_depth(x_gt, save_dir='test/x_gt.png')

depth_texture = x_gt.reshape(W,H,4)[...,2]
x = x.reshape(W,H,4)
depth = np.zeros([W, H])
depth_test = np.zeros([W, H])
depth_err = np.zeros([W, H])
bias = 0.05
for i in range(W):
    for j in range(H):
        w,h,z_cam = x[i,j,0]+1, x[i,j,1]+1, x[i,j,2]
        w,h = float(w), float(h)
        w,h = round(w*W/2), round(h*H/2)
        if w>W-1 or h>H-1 or w<0 or h<0:
            depth_test[i,j] = 1
            continue
        if abs(depth_texture[H-h-1, w] - z_cam) > bias:
        # if z_cam < depth_texture[H-h-1, w] - bias:
            depth_test[i,j] = 0
        else:
            depth_test[i,j] = 1
        
        depth_err[i,j] = abs(depth_texture[H-h-1, w] - z_cam)
        depth[i,j]  = depth_texture[H-h-1, w]

def composite(seg, rgb):
    seg_g = (seg==2).astype(float) # depth test pass
    seg_r = (seg==1).astype(float) # depth test fail
    seg_b = (seg==0).astype(float) # depth test out of range
    seg_rgb = np.stack([seg_r, seg_g, seg_b], axis=2) * 0.5
    return rgb+seg_rgb

rgb_comp = composite(depth_test, rgb_cam)
plot([depth],1,1,'test/depth.png')
rgb_vis = 0.8*rgb_cam + 0.2*rgb_cam*depth_test[...,None]
rgb_vis = rgb_vis + (rgb_vis==0)

plot([rgb_comp, depth_test, depth_err, rgb_light, rgb_cam, rgb_vis], 2, 3, save_dir='test/light1.png')


        # z_light = depth_texture[i][j]
        # z_cam = 

# plot_depth(pts_cam, save_dir='test/cam_world.png')
# plot_depth(pts_light, save_dir='test/light_world.png')

# pts_cam_gt = pts_cam @ w2c_cam.T @ proj.T
# # pts_cam_gt = pts_cam @ w2c_cam.T
# plot_depth(pts_cam_gt, save_dir='test/cam_gt.png')

# pts_light_gt = pts_light @ w2c_light.T @ proj.T
# z_gt = pts_light_gt[..., 2]
# print('z_gt',z_gt.shape)

# plot_depth(pts_light_gt, save_dir='test/light_gt.png')

# pts_light = pts_cam @ w2c_light.T @ proj.T
# pts_light = pts_light 

# # normalizer = pts_light[..., 3, None]
# # pts_light = pts_light/normalizer
# # pts_light = pts_light.reshape(W,H,4)

# depth_test = np.zeros([W, H])
# bias = 0.3
# for j in range(W):
#     for i in range(H):
#         pts_cam_gt = pts_cam.reshape(W, H, 4)
#         x,y = pts_cam_gt[i,j,:][0], pts_cam_gt[i,j,:][1]
#         z = pts_light[i,j,:][2]
#         # pt = pts_light[i,j,:]
#         # x,y,z = pt[0], pt[1], pt[2]
#         if abs(x)>1 or abs(y)>1:
#             depth_test[i][j] = 0
#             continue
#         x_idx, y_idx = int(x), int(y)
#         if abs(z-z_gt[x_idx][y_idx]) < bias:
#             depth_test[i][j] = 1
#         else:
#             depth_test[i][j] = 2

# def composite(seg, rgb):
#     seg_g = (seg==1).astype(float) # depth test pass
#     seg_r = (seg==2).astype(float) # depth test fail
#     seg_b = (seg==0).astype(float) # depth test out of range

#     seg_rgb = np.stack([seg_r, seg_g, seg_b], axis=2) * 0.5
#     return rgb+seg_rgb

# rgb_comp = composite(depth_test, rgb_cam)
# plot([rgb_comp, rgb_light], 1, 2, save_dir='test/light.png')
# plot_depth(pts_light, clip=True, save_dir='test/light.png')

