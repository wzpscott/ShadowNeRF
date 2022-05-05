from matplotlib.pyplot import vlines
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from encoding import *

class Rays():
    def __init__(self, img_gt, c2w, W, H, focal, near, far, device='cpu'):
        pass
    
    def batchify(self, batch_rays_class, random, clip):
        pass

    @staticmethod
    def convert_to_rays(c2w, W, H, focal):
        '''
        get o(rigins) and d(estinations) of rays
        '''
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i), torch.zeros_like(i)], -1) # [W, H, 4]
        rays_d = dirs @ c2w.T 
        rays_o = c2w[:,-1].expand(rays_d.shape)
        return rays_o, rays_d

    @staticmethod
    def get_view_matrix(c2w):
        '''
        get view matrix, or world-to-camera matrix
        '''
        return torch.linalg.inv(c2w)

    @staticmethod
    def get_perspective_matrix(W, H, near, far, focal):
        '''
        perspective projection matrix
        '''
        mat = torch.Tensor([
                [2*focal/W, 0, 0, 0],
                [0, 2*focal/H, 0, 0],
                [0, 0, -far/(far-near), -2*far*near/(far-near)],
                [0, 0, -1, 0]
            ])
        return mat

class BatchRays():
    def __init__(self, rays_o, rays_d, rgb_gt, W, H, focal, near, far, device='cpu'):
        pass
    def sample(self, N_samples, random=False):
        pass
    def encode(self, L_x, L_dir):
        pass
    def decode(self, model, **kwargs):
        pass
    def volume_render(self, render_list):
        pass

    @staticmethod
    def uniform_sample(N_samples, near, far, batch_size, random):
        t_vals = torch.linspace(0., 1., steps=N_samples) # [N_samples]
        z_vals = near * (1.-t_vals) + far * (t_vals) 
        z_vals = z_vals.unsqueeze(0) 
        z_vals = z_vals.expand([batch_size, N_samples]) # [N_rays, N_samples]
        if random:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand
    
        return z_vals


class CamRays(Rays):
    def __init__(self, img_gt, c2w, c2w_light, i_light, W, H, focal, near, far, device='cpu'):
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        img_gt, c2w, c2w_light = torch.Tensor(img_gt), torch.Tensor(c2w), torch.Tensor(c2w_light)

        # get o(rigins) and d(estinations) of rays
        rays_o, rays_d = self.convert_to_rays(c2w, W, H, focal)
        self.o = rays_o.reshape(-1,4).to(device) # [W*H,4], world coordinate
        self.d = rays_d.reshape(-1,4).to(device) # [W*H,4], world coordinate

        # store other useful attributes
        self.img_gt = img_gt.reshape(-1,3).to(device)
        self.cam_pos = c2w[:,-1].to(device)
        self.light_pos = c2w_light[:,-1].to(device)
        self.view_mat = self.get_view_matrix(c2w).to(device) # view matrix
        self.proj_mat = self.get_perspective_matrix(W, H, near, far, focal).to(device) # perspective projection matrix
        self.near, self.far, self.W, self.H, self.focal = near, far, W, H, focal
        self.i_light = i_light

    def batchify(self, batch_size, random=True, crop=False):
        idx = list(range(self.W * self.H))
        if crop:
            idx_crop = []
            for i in idx:
                w, h = i//self.W, i%self.W
                if w > self.W/4 and w < self.W/4*3 and h > self.H/4 and h < self.H/4*3:
                    idx_crop.append(i)
            idx = idx_crop

        if random:
            np.random.shuffle(idx)

        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            rays_o, rays_d = self.o[batch_idx], self.d[batch_idx]
            rgb_gt = self.img_gt[batch_idx]

            yield BatchCamRays(rays_o, rays_d, rgb_gt, self.cam_pos, self.light_pos, self.near, self.far, self.W, self.H, self.focal)

class BatchCamRays(BatchRays):
    def __init__(self, rays_o, rays_d, rgb_gt, cam_pos, light_pos, near, far, W, H, focal):
        self.batch_size = rays_o.shape[0]
        self.o = rays_o # [batch_size, 4]
        self.d = rays_d # [batch_size, 4]
        self.rgb_gt = rgb_gt # [batch_size, 3]
        self.cam_pos = cam_pos
        self.light_pos = light_pos
        self.near, self.far, self.W, self.H, self.focal = near, far, W, H, focal

    def sample(self, N_samples, random=False):
        z_vals = self.uniform_sample(N_samples, self.near, self.far, self.batch_size, random)
        pts = self.o[..., None, :] + self.d[..., None, :] *  z_vals[..., :, None]  # [N_rays, N_samples, 4], last dim: x,y,z,1
        self.pts = pts
        self.z_vals = z_vals

    def encode(self, L_x, L_dir):
        pts = self.pts[...,:3] # [N_rays, N_samples, 3], positions of sampled points
        dirs = self.d[...,None,:3].expand_as(pts) # [N_rays, N_samples, 3], view directions of sampled points
        pts_encoded, dirs_encoded = frequency_encode(pts, L_x), frequency_encode(dirs, L_dir)
        self.pts_encoded, self.dirs_encoded = pts_encoded, dirs_encoded

    def decode(self, model, **kwargs):
        pts_properties = model(self.pts_encoded, self.dirs_encoded, self.pts[...,:3], self.d[...,None,:3], \
            self.cam_pos, self.light_pos, **kwargs)
        self.pts_properties = pts_properties
        return pts_properties

    def shadow_mapping(self, light_rays, rays=None):
        def norm(x):
            N = x[:,3].unsqueeze(-1)
            return x/N

        def get_terminate_pts(rays_o, rays_d, depth):
            rays_d = F.normalize(rays_d, dim=-1)
            depth = depth.reshape(-1,1)
            # depth[:,0] = depth[:,0].clip(-6,-2)
            # depth[:,0] = depth[:,0].clip(2,6)
            pts = rays_o + depth * rays_d
            assert pts.shape[-1] == 4
            return pts

        # depth_cam, depth_light = self.pixel_properties['depth'], light_rays.depth_map
        depth_cam, depth_light = self.pixel_properties['depth'], light_rays.pixel_properties['depth']
        o_cam, d_cam = self.o, self.d
        o_light, d_light = light_rays.o, light_rays.d

        pts_cam = get_terminate_pts(o_cam, d_cam, depth_cam)
        pts_light = get_terminate_pts(o_light, d_light, depth_light)

        view_light, proj_light = light_rays.view_mat, light_rays.proj_mat

        x = pts_cam @ view_light.T
        x = x @ proj_light.T
        x = norm(x)

        x_gt = pts_light @ view_light.T
        # x_gt[:, 2] = x_gt[:, 2].clip(-6,-2)
        x_gt = x_gt @ proj_light.T
        x_gt = norm(x_gt)

        W,H = self.W, self.H
        depth_test = torch.zeros(x.shape[0])
        depth_err = torch.zeros(x.shape[0])
        depth_light = x_gt.reshape(W,H,4)[...,2]
        bias = 0.05

        for i in range(x.shape[0]):
            w,h,z_cam = x[i,0]+1, x[i,1]+1, x[i,2]

            w,h = float(w), float(h)
            w,h = round(w*W/2), round(h*H/2)

            if w>W-1 or h>H-1 or w<0 or h<0:
                depth_test[i] = 1
                depth_err[i] = 0.5
                continue
            
            if abs(depth_light[H-h-1, w] - z_cam) > bias:
                depth_test[i] = 0
            else:
                depth_test[i] = 1
            depth_err[i] = abs(depth_light[H-h-1, w] - z_cam)
        return x, x_gt, depth_test, depth_err


    def volume_render(self, render_list):
        dists = self.z_vals[...,1:] - self.z_vals[...,:-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], 
            dim=-1)  # [N_rays, N_samples]

        dists = (dists * torch.norm(self.d[..., None, :3], dim=-1)).unsqueeze(-1)

        alpha, rgb = self.pts_properties['alpha'], self.pts_properties['rgb']
        alpha = 1.-torch.exp(-F.relu(alpha) * dists)

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones(alpha.shape[0], 1, 1), 1.-alpha + 1e-10], 1), 1)[:, :-1, :]

        render_dict = {'depth':self.z_vals.unsqueeze(-1), 'rgb':rgb}

        for attr in render_list:
            render_dict[attr] = self.pts_properties[attr]
        pixel_properties = {}
        for k,v in render_dict.items():
            pixel_properties[k] = torch.sum(weights * v, 1)
        self.pixel_properties = pixel_properties
        return pixel_properties


class LightRays(Rays):
    def __init__(self, c2w, W, H, focal, near, far, device='cpu'):
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        c2w = torch.Tensor(c2w)

        # get o(rigins) and d(estinations) of rays
        rays_o, rays_d = self.convert_to_rays(c2w, W, H, focal)
        self.o = rays_o.reshape(-1,4).to(device) # [W*H,4], world coordinate
        self.d = rays_d.reshape(-1,4).to(device) # [W*H,4], world coordinate

        # store useful attributes
        self.pos = c2w[:,-1].to(device)
        self.view_mat = self.get_view_matrix(c2w).to(device) # view matrix
        self.proj_mat = self.get_perspective_matrix(W, H, near, far, focal).to(device) # perspective projection matrix
        self.near, self.far, self.W, self.H, self.focal = near, far, W, H, focal

        # store depth map
        self.depth_map = torch.zeros(W*H)

    def batchify(self, batch_size):
        idx = list(range(self.W * self.H))

        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            rays_o, rays_d = self.o[batch_idx], self.d[batch_idx]

            yield BatchLightRays(
                rays_o, rays_d, self.pos, self.near, self.far, self.W, self.H, self.focal)

    def update_depth_map(self, batch_size, num_samples, L_x, L_dir, model):
        for i, rays_batch in enumerate(self.batchify(batch_size)):
            rays_batch.sample(num_samples)
            rays_batch.encode(L_x, L_dir)
            rays_batch.decode(model)
            rays_batch.volume_render()
            depth = rays_batch.pixel_properties['depth'].reshape(-1)
            self.depth_map[i*batch_size:(i+1)*batch_size] = depth

class BatchLightRays(BatchRays):
    def __init__(self, rays_o, rays_d, pos, near, far, W, H, focal):
        self.batch_size = rays_o.shape[0]
        self.o = rays_o # [batch_size, 4]
        self.d = rays_d # [batch_size, 4]
        self.pos = pos
        self.near, self.far, self.W, self.H, self.focal = near, far, W, H, focal

    def sample(self, N_samples, random=False):
        z_vals = self.uniform_sample(N_samples, self.near, self.far, self.batch_size, random)
        pts = self.o[..., None, :] + self.d[..., None, :] *  z_vals[..., :, None]  # [N_rays, N_samples, 4], last dim: x,y,z,1
        self.pts = pts
        self.z_vals = z_vals

    def encode(self, L_x, L_dir):
        pts = self.pts[...,:3] # [N_rays, N_samples, 3], positions of sampled points
        dirs = self.d[...,None,:3].expand_as(pts) # [N_rays, N_samples, 3], view directions of sampled points
        pts_encoded, dirs_encoded = frequency_encode(pts, L_x), frequency_encode(dirs, L_dir)
        self.pts_encoded, self.dirs_encoded = pts_encoded, dirs_encoded

    def decode(self, model, **kwargs):
        pts_properties = model(self.pts_encoded, self.dirs_encoded, self.pts[...,:3], self.d[...,None,:3], \
            self.pos, **kwargs)
        self.pts_properties = pts_properties
        return pts_properties

    def volume_render(self, render_list=[]):
        dists = self.z_vals[...,1:] - self.z_vals[...,:-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], 
            dim=-1)  # [N_rays, N_samples]

        dists = (dists * torch.norm(self.d[..., None, :3], dim=-1)).unsqueeze(-1)

        alpha, rgb = self.pts_properties['alpha'], self.pts_properties['rgb']
        alpha = 1.-torch.exp(-F.relu(alpha) * dists)

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones(alpha.shape[0], 1, 1), 1.-alpha + 1e-10], 1), 1)[:, :-1, :]

        render_dict = {'depth':self.z_vals.unsqueeze(-1), 'rgb':rgb}
        for attr in render_list:
            render_dict[attr] = self.pts_properties[attr]
        pixel_properties = {}
        for k,v in render_dict.items():
            pixel_properties[k] = torch.sum(weights * v, 1)
        self.pixel_properties = pixel_properties
        return pixel_properties