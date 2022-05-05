from tabnanny import verbose
import torch
import numpy as np
import matplotlib.pyplot as plt
from rays import Rays
import os, os.path as osp
from tqdm import tqdm

def img2mse(x, y):
    x = x.squeeze()
    y = y.squeeze()
    return torch.mean((x - y) ** 2)

def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)

@torch.no_grad()
def eval_image(rays, L_x, L_dir, model, batch_size=1024, num_samples=128, light_rays=None):
    rgb_pred = []
    depth_pred = []
    normal_pred = []
    ambient_pred = []
    diffuse_pred = []
    specular_pred = []
    shadow_pred = []

    for rays_batch in rays.batchify(batch_size, random=False):
        rays_batch.sample(num_samples)
        rays_batch.encode(L_x, L_dir)
        rays_batch.decode(model)
        rays_batch.volume_render(render_list=['normal', 'ambient', 'diffuse', 'specular'])

        depth_test, depth_err = rays_batch.shadow_mapping(
            light_rays.view_mat, light_rays.proj_mat, rays_batch.pixel_properties['depth'], light_rays.depth_map)
        # depth_test, depth_err = rays_batch.shadow_mapping(
        #     rays.view_mat, rays.proj_mat, rays_batch.pixel_properties['depth'], light_rays.depth_map)


        rgb_pred.append(rays_batch.pixel_properties['rgb'].cpu())
        depth_pred.append(rays_batch.pixel_properties['depth'].cpu())
        normal_pred.append(rays_batch.pixel_properties['normal'].cpu())
        ambient_pred.append(rays_batch.pixel_properties['ambient'].cpu())
        diffuse_pred.append(rays_batch.pixel_properties['diffuse'].cpu())
        specular_pred.append(rays_batch.pixel_properties['specular'].cpu())

        shadow_pred.append(depth_err.cpu())

    rgb_pred = torch.cat(rgb_pred,dim=0) # [W*H, 3]
    depth_pred = torch.cat(depth_pred,dim=0) # [W*H, 1]

    normal_pred = torch.cat(normal_pred,dim=0)
    normal_pred = torch.cat([normal_pred, torch.zeros(normal_pred.shape[0], 1).cpu()], dim=-1)
    normal_pred = normal_pred @ rays.view_mat.cpu().T @ rays.proj_mat.cpu().T
    normal_pred = 0.5*normal_pred + 0.5
    normal_pred = normal_pred[:,:3]


    ambient_pred = torch.cat(ambient_pred,dim=0)
    diffuse_pred = torch.cat(diffuse_pred,dim=0)
    specular_pred = torch.cat(specular_pred,dim=0)

    shadow_pred = torch.cat(shadow_pred,dim=0)
    return rgb_pred, depth_pred, normal_pred, ambient_pred, diffuse_pred, specular_pred, shadow_pred

@torch.no_grad()
def eval_shadow(rays, light_rays, L_x, L_dir, model, batch_size=1024, num_samples=128):
    depth_cam = []
    rgb_cam = []
    depth_light = []
    rgb_light = []

    for rays_batch in rays.batchify(batch_size, random=False):
        rays_batch.sample(num_samples)
        rays_batch.encode(L_x, L_dir)
        rays_batch.decode(model)
        rays_batch.volume_render(render_list=[])

        rgb_cam.append(rays_batch.pixel_properties['rgb'].cpu())
        depth_cam.append(rays_batch.pixel_properties['depth'].cpu())

        # x, x_gt, depth_test, depth_err = rays_batch.shadow_mapping(light_rays,rays)
        # shadow.append(depth_err.cpu())
        # xs.append(x.cpu())
        # x_gts.append(x_gt.cpu())

    
    for rays_batch in light_rays.batchify(batch_size):
        rays_batch.sample(num_samples)
        rays_batch.encode(L_x, L_dir)
        rays_batch.decode(model)
        rays_batch.volume_render(render_list=[])

        rgb_light.append(rays_batch.pixel_properties['rgb'].cpu())
        depth_light.append(rays_batch.pixel_properties['depth'].cpu())


    depth_cam = torch.cat(depth_cam,dim=0)
    rgb_cam = torch.cat(rgb_cam,dim=0)
    depth_light = torch.cat(depth_light,dim=0)
    rgb_light = torch.cat(rgb_light,dim=0)
    return depth_cam, depth_light, rgb_cam, rgb_light

def plot(imgs, m, n, save_dir=''):
    for i, img in enumerate(imgs):
        img = np.array(img)
        plt.subplot(m, n, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(save_dir)

def write(s, log_dir):
    tqdm.write(s)
    with open(log_dir, 'a') as fp:
        fp.write(s+'\n')
