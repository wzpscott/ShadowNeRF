import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from encoding import frequency_encode

class Rays():
    def __init__(self, images, c2w, W, H, focal, near, far, c2w_light=None, device='cpu'):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        images, c2w = torch.Tensor(images), torch.Tensor(c2w)
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i), torch.zeros_like(i)], -1) # [W, H, 4]
        rays_d = dirs @ c2w.T 
        # rays_d = F.normalize(rays_d, dim=-1)
        rays_o = c2w[:,-1].expand(rays_d.shape)

        self.o = rays_o.reshape(-1,4).to(device) # [W*H,4], world coordinate
        self.d = rays_d.reshape(-1,4).to(device) # [W*H,4], world coordinate

        self.cam_pos = c2w[:,-1].to(device)
        if c2w_light is not None:
            c2w_light = torch.Tensor(c2w_light)
            self.light_pos = c2w_light[:,-1].to(device)
        else:
            self.light_pos = None

        self.view_mat = torch.linalg.inv(c2w).to(device) # view matrix, or world-to-camera matrix
        self.proj_mat = torch.Tensor([
            [2*focal/W, 0, 0, 0],
            [0, 2*focal/H, 0, 0],
            [0, 0, -far/(far-near), -2*far*near/(far-near)],
            [0, 0, -1, 0]
        ]).to(device) # perspective projection matrix

        self.images = images.reshape(-1,3).to(device) # store 
        self.near, self.far, self.W, self.H, self.focal = near, far, W, H, focal # store everything else
    
    def sample_along_rays(self, N_samples, random=False):
        t_vals = torch.linspace(0., 1., steps=N_samples) # [N_samples]
        z_vals = self.near * (1.-t_vals) + self.far * (t_vals) 
        z_vals = z_vals.unsqueeze(0) 
        z_vals = z_vals.expand([self.o.shape[0], N_samples]) # [N_rays, N_samples]
        if random:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand
        pts = self.o[..., None, :] + self.d[..., None, :] *  z_vals[..., :, None]  # [N_rays, N_samples, 4], last dim: x,y,z,1
        self.pts = pts
        self.z_vals = z_vals
        return pts

    # def query(self, encode_fn, decode_fn):
    #     if 'pts' not in self.__dict__.keys():
    #         raise ValueError('do not have attr "pts", need to execute "sample_along_rays" first')
        
    #     pts = self.pts[...,:3] # [N_rays, N_samples, 3], positions of sampled points
    #     dirs = self.d[...,None,:3].expand_as(pts) # [N_rays, N_samples, 3], view directions of sampled points

    #     pts_encoded, dirs_encoded = encode_fn(pts, dirs)
    #     raw_properties = decode_fn(pts_encoded, dirs_encoded, cam_pos=self.cam_pos, light_pos=self.light_pos) # a dict stores {property_name: X^[N_rays, N_samples, _]}
    #     self.raw_properties = raw_properties
    #     return raw_properties
    def encode(self, L_x, L_dir, encode_fn):
        if encode_fn == 'freq':
            pts = self.pts[...,:3] # [N_rays, N_samples, 3], positions of sampled points
            dirs = self.d[...,None,:3].expand_as(pts) # [N_rays, N_samples, 3], view directions of sampled points
            pts_encoded, dirs_encoded = frequency_encode(pts, L_x), frequency_encode(dirs, L_dir)
            self.pts_encoded, self.dirs_encoded = pts_encoded, dirs_encoded
        else:
            raise NotImplementedError()

    def decode(self, model):
        raw_properties = model(self.pts_encoded, self.dirs_encoded, self.pts[...,:3], self.d[...,None,:3], cam_pos=self.cam_pos, light_pos=self.light_pos)
        self.raw_properties = raw_properties
        return raw_properties

    def volume_render(self, verbose=False):
        if 'raw_properties' not in self.__dict__.keys():
            raise ValueError('do not have attr "raw_properties", need to execute "encode" and "decode" first')

        dists = self.z_vals[...,1:] - self.z_vals[...,:-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], 
            dim=-1)  # [N_rays, N_samples]

        dists = (dists * torch.norm(self.d[..., None, :3], dim=-1)).unsqueeze(-1)

        alpha, rgb = self.raw_properties['alpha'], self.raw_properties['rgb']
        alpha = 1.-torch.exp(-F.relu(alpha) * dists)

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones(alpha.shape[0], 1, 1), 1.-alpha + 1e-10], 1), 1)[:, :-1, :]

        properties = {}
        if not verbose:
            render_dict = {'depth':weights, 'acc': alpha, 'rgb':rgb}
        else:
            normal, ambient, diffuse, specular = self.raw_properties['normal'], self.raw_properties['ambient'], self.raw_properties['diffuse'], self.raw_properties['specular']
            render_dict = {'depth':weights, 'acc': alpha, 'rgb':rgb,
            'normal':normal, 'ambient':ambient, 'diffuse':diffuse, 'specular':specular}
        for k,v in render_dict.items():
            properties[k] = torch.sum(weights * v, 1)
        self.properties = properties
        return properties
        
    def batchify(self, batch_size, random=True, crop=False):
        idx = list(range(self.W * self.H))
        if crop:
            idx_crop = []
            for i in idx:
                w, h = i//W, i%W
                if w > W/4 and w < W/4*3 and h > H/4 and h < H/4*3:
                    idx_crop.append(i)
            idx = idx_crop

        if random:
            np.random.shuffle(idx)

        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            rays_o, rays_d = self.o[batch_idx], self.d[batch_idx]
            rgb_gt = self.images[batch_idx]
            yield RaysBatch(rays_o, rays_d, rgb_gt, self.cam_pos, self.light_pos, self.near, self.far, self.W, self.H, self.focal)

    def peek(self):
        print(f'attrs: {(self.__dict__).keys()}')
        print(f'rays_o: shape {self.o.shape}, example {self.o[0].data}')
        print(f'rays_d: shape {self.d.shape}, example {self.d[0].data}')
        if 'pts' in self.__dict__:
            print(f'pts: shape {self.pts.shape}, example {self.pts[0,0,:].data}')
        if 'raw_properties' in self.__dict__:
            for prop in self.raw_properties:
                print(f'{prop}: shape {self.raw_properties[prop].shape}, \
                    example {self.raw_properties[prop][0,0,:].data}')
        if 'properties' in self.__dict__:
            for prop in self.properties:
                print(f'{prop}: shape {self.properties[prop].shape}, \
                    example {self.properties[prop][0,:].data}')

class RaysBatch(Rays):
    def __init__(self, rays_o, rays_d, rgb_gt, cam_pos, light_pos, near, far, W, H, focal):
        self.batch_size = rays_o.shape[0]
        self.o = rays_o # [batch_size, 4]
        self.d = rays_d # [batch_size, 4]
        self.rgb_gt = rgb_gt
        self.cam_pos, self.light_pos = cam_pos, light_pos
        self.near, self.far, self.W, self.H, self.focal = near, far, W, H, focal

if __name__ == '__main__':
    c2w = torch.rand(4,4)
    c2w_light = torch.rand(4,4)
    W, H = 100, 100
    images = torch.rand(W,H,3)
    focal = 1
    near, far = 2,6
    rays = Rays(images, c2w, W, H, focal, near, far, c2w_light)
    rays.peek()

    # ray_batches = rays.batchify(128)
    # for ray_batch in ray_batches:
    #     ray_batch.printf()
    # rays.sample_along_rays(16)