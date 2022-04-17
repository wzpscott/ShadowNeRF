import torch
import numpy as np
import matplotlib.pyplot as plt
from rays import Rays

def img2mse(x, y):
    x = x.squeeze()
    y = y.squeeze()
    return torch.mean((x - y) ** 2)

def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)

@torch.no_grad()
def eval_image(rays, encode_fn, decode_fn, batch_size=1024, num_samples=128):
    rgb_pred = []
    depth_pred = []
    for rays_batch in rays.batchify(batch_size, random=False):
        rays_batch.sample_along_rays(num_samples)
        rays_batch.query(encode_fn, decode_fn)
        rays_batch.volume_render()
        rgb_pred.append(rays_batch.properties['rgb'].cpu())
        depth_pred.append(rays_batch.properties['depth'].cpu())

    rgb_pred = torch.cat(rgb_pred,dim=0) # [W*H, 3]
    depth_pred = torch.cat(depth_pred,dim=0) # [W*H, 1]
    assert len(depth_pred.shape) == 2 
    assert depth_pred.shape[1] == 1

    return rgb_pred, depth_pred

def plot(imgs, m, n, save_dir=''):
    for i, img in enumerate(imgs):
        img = np.array(img)
        plt.subplot(m, n, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(save_dir)
