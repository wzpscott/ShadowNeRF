import os.path as osp
import numpy as np
import imageio 
import json
import cv2

# def load_from_json(transforms, include_light=True, resize_factor=1):
#     imgs = []
#     cam_poses = []
#     light_poses = []
#     i_light_poses = []

#     frames = transforms['frames']
#     for i, frame in enumerate(frames):
#         img = imageio.imread(frame['file_path'])
#         cam_pose = np.array(frame['transform_matrix_cam'])
#         if include_light:     
#             light_pose = np.array(frame['transform_matrix_light'])

#         imgs.append(img)
#         cam_poses.append(cam_pose)
#         if include_light:
#             light_poses.append(light_pose)
#             i_light_poses.append(i%10)

#     imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
#     cam_poses = np.array(cam_poses).astype(np.float32)
#     light_poses = np.array(light_poses).astype(np.float32)

#     H, W = imgs[0].shape[:2]
#     camera_angle_x = float(transforms['camera_angle_x'])
#     focal = .5 * W / np.tan(.5 * camera_angle_x)

#     if resize_factor != 1:
#         H = H//resize_factor
#         W = W//resize_factor
#         focal = focal/resize_factor

#         imgs_resized = np.zeros((imgs.shape[0], H, W, 4))
#         for i, img in enumerate(imgs):
#             imgs_resized[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
#         imgs = imgs_resized

#     data = {
#         'imgs':imgs,
#         'cam_poses': cam_poses,
#         'light_poses':light_poses,
#         'i_light_poses':i_light_poses, 
#         'hwf':[H, W, focal]
#     }
#     return data
        
# def load_blender_data(basedir, include_light=True, resize_factor=2):
#     with open(osp.join(basedir, 'transforms_train.json'), 'r') as fp:
#         transforms_train = json.load(fp)
#     with open(osp.join(basedir, 'transforms_test.json'), 'r') as fp:
#         transforms_test = json.load(fp)
    
#     data_train =load_from_json(transforms_train, include_light, resize_factor)
#     data_test = load_from_json(transforms_test, include_light, resize_factor)

#     images = np.concatenate([data_train['imgs'], data_test['imgs']], axis=0)
#     cam_poses = np.concatenate([data_train['cam_poses'], data_test['cam_poses']], axis=0)
#     light_poses = np.concatenate([data_train['light_poses'], data_test['light_poses']], axis=0)
#     i_light_poses = data_train['i_light_poses'] + data_test['i_light_poses']

#     N_train = data_train['imgs'].shape[0]
#     N_test = data_test['imgs'].shape[0]
#     i_splits = [[i for i in range(N_train)],
#                 [i for i in range(N_train, N_train + N_test//2)],
#                 [i for i in range(N_train + N_test//2, N_train + N_test)]]

#     hwf = data_train['hwf']

#     return images, cam_poses, light_poses, i_light_poses, hwf, i_splits


def load_from_json(transforms, include_light=True, resize_factor=1):
    imgs = []
    cam_poses = []
    light_poses = []
    i_light_poses = []

    frames = transforms['frames']
    for i, frame in enumerate(frames[:3  ]):
        img = imageio.imread(frame['file_path'])
        cam_pose = np.array(frame['transform_matrix_cam'])
        if include_light:     
            light_pose = np.array(frame['transform_matrix_light'])

        imgs.append(img)
        cam_poses.append(cam_pose)
        if include_light:
            light_poses.append(light_pose)
            i_light_poses.append(i%10)

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    cam_poses = np.array(cam_poses).astype(np.float32)
    light_poses = np.array(light_poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(transforms['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if resize_factor != 1:
        H = H//resize_factor
        W = W//resize_factor
        focal = focal/resize_factor

        imgs_resized = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_resized[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_resized

    data = {
        'imgs':imgs,
        'cam_poses': cam_poses,
        'light_poses':light_poses,
        'i_light_poses':i_light_poses, 
        'hwf':[H, W, focal]
    }
    return data
        
def load_blender_data(basedir, include_light=True, resize_factor=2):
    with open(osp.join(basedir, 'transforms_train.json'), 'r') as fp:
        transforms_train = json.load(fp)
    with open(osp.join(basedir, 'transforms_test.json'), 'r') as fp:
        transforms_test = json.load(fp)
    
    data_train =load_from_json(transforms_train, include_light, resize_factor)
    data_test = load_from_json(transforms_test, include_light, resize_factor)

    images = np.concatenate([data_train['imgs'], data_test['imgs']], axis=0)
    cam_poses = np.concatenate([data_train['cam_poses'], data_test['cam_poses']], axis=0)
    light_poses = np.concatenate([data_train['light_poses'], data_test['light_poses']], axis=0)
    i_light_poses = data_train['i_light_poses'] + data_test['i_light_poses']

    N_train = data_train['imgs'].shape[0]
    N_test = data_test['imgs'].shape[0]
    i_splits = [[i for i in range(N_train)],
                [i for i in range(N_train, N_train + N_test//2)],
                [i for i in range(N_train + N_test//2, N_train + N_test)]]

    hwf = data_train['hwf']

    return images, cam_poses, light_poses, i_light_poses, hwf, i_splits