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
def eval_image(rays, L_x, L_dir, model, batch_size=1024, num_samples=128):
    rgb_pred = []
    depth_pred = []
    normal_pred = []
    ambient_pred = []
    diffuse_pred = []
    specular_pred = []
    for rays_batch in rays.batchify(batch_size, random=False):
        rays_batch.sample_along_rays(num_samples)
        rays_batch.encode(L_x, L_dir, encode_fn='freq')
        rays_batch.decode(model)
        rays_batch.volume_render(verbose=True)

        rgb_pred.append(rays_batch.properties['rgb'].cpu())
        depth_pred.append(rays_batch.properties['depth'].cpu())
        normal_pred.append(rays_batch.properties['normal'].cpu())
        ambient_pred.append(rays_batch.properties['ambient'].cpu())
        diffuse_pred.append(rays_batch.properties['diffuse'].cpu())
        specular_pred.append(rays_batch.properties['specular'].cpu())

    rgb_pred = torch.cat(rgb_pred,dim=0) # [W*H, 3]
    depth_pred = torch.cat(depth_pred,dim=0) # [W*H, 1]
    normal_pred = torch.cat(normal_pred,dim=0)
    normal_pred = 0.5*normal_pred + 0.5
    ambient_pred = torch.cat(ambient_pred,dim=0)
    diffuse_pred = torch.cat(diffuse_pred,dim=0)
    specular_pred = torch.cat(specular_pred,dim=0)
    assert len(depth_pred.shape) == 2 
    assert depth_pred.shape[1] == 1

    return rgb_pred, depth_pred, normal_pred, ambient_pred, diffuse_pred, specular_pred

def plot(imgs, m, n, save_dir=''):
    for i, img in enumerate(imgs):
        img = np.array(img)
        plt.subplot(m, n, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(save_dir)

def write(s, fp):
    tqdm.write(s)
    fp.write(s+'\n')
