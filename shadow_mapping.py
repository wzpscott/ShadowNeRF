import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import load_blender_data
from models import NePhong
from rays import *
from utils import *

import open3d as o3d

datadir = './data/nerf_synthetic_relight/hotdog'
images, cam_poses, light_poses, i_light_poses, hwf, i_splits = load_blender_data(
                                                                datadir, include_light=True, resize_factor=20)
images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:]) # white background

BATCH_SIZE, NUM_SAMPLES = 1024, 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
near = 2.
far = 6.
H, W, focal = hwf
L_x, L_dir = 10, 4

model = NePhong(x_dim=2*L_x*3+3, dir_dim=2*L_dir*3+3)
checkpoint = torch.load('/home/wzpscott/NeuralRendering/shadownerf/logs/hotdog_1/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(DEVICE)

i = 0
cam_rays = CamRays(images[i], cam_poses[i], light_poses[i], i_light_poses[i], W, H, focal, near, far,  DEVICE)
light_rays = LightRays(light_poses[i_light_poses[i]], W, H, focal, near, far, DEVICE)
# light_rays = CamRays(images[i+1], cam_poses[i+1], light_poses[i+1], i_light_poses[i+1], W, H, focal, near, far,  DEVICE)

depth_cam, depth_light, rgb_cam, rgb_light = eval_shadow(
                                                    cam_rays, light_rays, L_x, L_dir, model, BATCH_SIZE, NUM_SAMPLES)

depth_cam, depth_light, rgb_cam, rgb_light = depth_cam.cpu(), depth_light.cpu(), rgb_cam.cpu(), rgb_light.cpu()
depth_cam, depth_light, rgb_cam, rgb_light = depth_cam.reshape(W,H,-1), depth_light.reshape(W,H,-1), rgb_cam.reshape(W,H,-1), rgb_light.reshape(W,H,-1)
# print(depth_cam.min(), depth_cam.max(), depth_cam.mean())
# print(depth_light.min(), depth_light.max(), depth_light.mean())
# raise ValueError()

# plot([rgb_cam, depth_cam, rgb_light, depth_light], m=2, n=2, save_dir='test.png')
# raise ValueError('ddd')


def norm(x):
    N = x[:,3].unsqueeze(-1)
    x[:, :2] /= N
    return x

def get_terminate_pts(rays_o, rays_d, depth):
    depth = depth.reshape(-1,1)
    # pts = rays_o + (near*(1-depth)+far*depth) * rays_d
    pts = rays_o + depth * rays_d
    return pts

def describe(x):
    print('shape:', x.shape)
    print(f'x: min:{x[:,0].min()} max:{x[:,0].max()}')
    print(f'y: min:{x[:,1].min()} max:{x[:,1].max()}')
    print(f'z: min:{x[:,2].min()} max:{x[:,2].max()}')
    print(f'norm: min:{x[:,3].min()} max:{x[:,3].max()}')

o_cam, d_cam = cam_rays.o.cpu(), cam_rays.d.cpu()
o_light, d_light = light_rays.o.cpu(), light_rays.d.cpu()

pts_cam = get_terminate_pts(o_cam, d_cam, depth_cam)
pts_light = get_terminate_pts(o_light, d_light, depth_light)

# pts_cam, pts_light = pts_cam[:, :3].numpy(), pts_light[:, :3].numpy()
# clrs_cam = np.tile(np.array([[1,0,0]]), [1600, 1])
# clrs_light = np.tile(np.array([[0,1,0]]), [1600, 1])

# pcd_cam = o3d.geometry.PointCloud()
# pcd_cam.points = o3d.utility.Vector3dVector(pts_cam)
# pcd_cam.colors = o3d.utility.Vector3dVector(clrs_cam)

# pcd_light = o3d.geometry.PointCloud()
# pcd_light.points = o3d.utility.Vector3dVector(pts_light)
# pcd_light.colors = o3d.utility.Vector3dVector(clrs_light)


# def get_frustrm(rays_o, rays_d, color=[0,0,1]):
#     o = rays_o[0,:3].unsqueeze(0).numpy()
#     rays_d = rays_d.reshape(H, W, 4)

#     d = np.stack([rays_d[0, 0, :3], rays_d[0, -1, :3], rays_d[-1, 0, :3], rays_d[-1, -1, :3]]).reshape(4,3)
#     d = o+d
#     pts = np.concatenate([o, d], axis=0)

#     frustrm = o3d.geometry.LineSet()
#     frustrm.points = o3d.utility.Vector3dVector(pts)
#     frustrm.colors = o3d.utility.Vector3dVector([color for i in range(8)])
#     frustrm.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,4],[4,3],[3,1]])

#     return frustrm

# cam_frustrm = get_frustrm(o_cam, d_cam)
# light_frustrm = get_frustrm(o_light, d_light)

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
# o3d.visualization.draw_geometries([pcd_cam, pcd_light, cam_frustrm, light_frustrm, frame])
# raise ValueError()

view_light, proj_light = light_rays.view_mat.cpu(), light_rays.proj_mat.cpu()

x = pts_cam @ view_light.T
x = x @ proj_light.T
x = norm(x)
# x = x-3
# x = x.clip(-6,-2)

x_gt = pts_light @ view_light.T
x_gt = x_gt @ proj_light.T
x_gt = norm(x_gt)

# describe(x)
# describe(x_gt)
# # describe(pts_light @ view_light.T @ proj_light.T)
# x, x_gt, rgb_cam, rgb_light = x.reshape(W,H,-1), x_gt.reshape(W,H,-1), rgb_cam.reshape(W,H,-1), rgb_light.reshape(W,H,-1)
# x, x_gt = x[...,2], x_gt[...,2]

# plot([rgb_cam, x, rgb_light, x_gt], m=2, n=2, save_dir='test.png')

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
            depth_test[i,j] = 0.5
            continue
        if abs(depth_texture[H-h-1, w] - z_cam) > bias:
            depth_test[i,j] = 0
        else:
            depth_test[i,j] = 1
        
        depth_err[i,j] = abs(depth_texture[H-h-1, w] - z_cam)
        depth[i,j]  = depth_texture[H-h-1, w]

describe(x.reshape(-1,4))
describe(x_gt)
x, x_gt, rgb_cam, rgb_light = x.reshape(W,H,-1), x_gt.reshape(W,H,-1), rgb_cam.reshape(W,H,-1), rgb_light.reshape(W,H,-1)
x, x_gt = x[...,2], x_gt[...,2]

plot([rgb_cam, x, rgb_light, x_gt, depth_err, depth_test], m=3, n=2, save_dir='test.png')