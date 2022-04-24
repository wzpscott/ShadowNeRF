import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, x_dim, dir_dim, D=8, W=256, skips=[4]):
        super().__init__()
        self.x_dim = x_dim
        self.dir_dim = dir_dim
        self.W = W
        self.D = D
        self.skips = skips

        self.x_linears = nn.ModuleList(
            [nn.Linear(x_dim, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + x_dim, W) for i in range(D-1)])

        self.dir_linears = nn.ModuleList([nn.Linear(dir_dim + W, W//2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

    def forward(self, x, dir):
        h = x
        for i, l in enumerate(self.x_linears):
            h = self.x_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)

        h = torch.cat([feature, dir], -1)

        for i, l in enumerate(self.dir_linears):
            h = self.dir_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)
        rgb = torch.sigmoid(rgb)
        # output = torch.cat([alpha, rgb], dim=-1)
        output = {'alpha':alpha, 'rgb':rgb}
        return output


class NePhong(nn.Module):
    def __init__(self, x_dim, dir_dim, D=8, W=256, skips=[4]):
        super().__init__()
        self.x_dim = x_dim
        self.dir_dim = dir_dim
        self.W = W
        self.D = D
        self.skips = skips

        self.x_linears = nn.ModuleList(
            [nn.Linear(x_dim, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + x_dim, W) for i in range(D-1)])

        self.dir_linears = nn.ModuleList([nn.Linear(dir_dim + W, W//2)])

        self.alpha_linear = nn.Linear(W, 1)
        self.shininess_linear = nn.Linear(W, 1)
        self.ambient_linear = nn.Linear(W, 3)
        self.normal_linear = nn.Linear(W, 3)

        self.feature_linear = nn.Linear(W, W)

        self.diffuse_linear = nn.Linear(W//2, 3)
        self.specular_linear = nn.Linear(W//2, 3)
    
    def forward(self, x_encoded, view_dir_encoded, x, view_dir, cam_pos, light_pos):
        h = x_encoded
        for i, l in enumerate(self.x_linears):
            h = self.x_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_encoded, h], -1)

        alpha = self.alpha_linear(h)
        shininess = self.shininess_linear(h)
        ambient = self.ambient_linear(h)
        normal = self.normal_linear(h)

        feature = self.feature_linear(h)

        h = torch.cat([feature, view_dir_encoded], -1)
        for i, l in enumerate(self.dir_linears):
            h = self.dir_linears[i](h)
            h = F.relu(h)

        diffuse = self.diffuse_linear(h)
        specular = self.specular_linear(h)

        diffuse = torch.sigmoid(diffuse)
        specular = torch.sigmoid(specular)
        ambient = torch.sigmoid(ambient)

        view_dir_ = F.normalize(view_dir, dim=-1)
        view_dir = x - cam_pos[:3].reshape(1,1,3) # [N_rays, N_samples, 3]
        view_dir = F.normalize(view_dir, dim=-1)      
    
        light_dir = x - light_pos[:3].reshape(1,1,3) # [N_rays, N_samples, 3]
        light_dir = F.normalize(light_dir, dim=-1)

        rgb = self.shade(diffuse, specular, ambient, shininess, normal, view_dir, light_dir, light_color=1)
        output = {
            'alpha': alpha, 
            'rgb': rgb, 
            'diffuse': diffuse, 
            'specular': specular, 
            'ambient': ambient, 
            'shininess': shininess, 
            'normal': normal
        }
        return output

    @staticmethod
    def shade(diffuse, specular, ambient, shininess, normal, view_dir, light_dir, light_color=1):

        normal = F.normalize(normal, dim=-1) # [N_rays, N_samples, 3]
        view_dir = F.normalize(view_dir) # # [N_rays, N_samples, 3]
        light_dir = F.normalize(light_dir) # [N_rays, N_samples, 3]

        # ambient
        ambient = light_color * ambient

        # diffuse
        diff = torch.max((normal*light_dir).sum(dim=-1,keepdim=True), torch.zeros_like(normal)) # [N_rays, N_samples, 3]*[N_rays, N_samples, 3]->[N_rays, N_samples, 1]
        diffuse = light_color * diff * diffuse

        #specular
        reflect_dir = view_dir - 2 * ((normal*light_dir).sum(dim=-1,keepdim=True)) * normal
        reflect_dir = F.normalize(reflect_dir, dim=-1)
        # print(reflect_dir[0,0,:])
        # raise ValueError
        spec = torch.max((view_dir*reflect_dir).sum(dim=-1,keepdim=True), torch.zeros(1))
        spec = torch.pow(spec+1e-5, shininess) # add 1e-5 to avoid numeric unstable
        specular = light_color * spec * specular
        rgb = 0.1 * ambient + diffuse + specular
        # rgb = 0.1*ambient + diffuse

        return rgb
        
        