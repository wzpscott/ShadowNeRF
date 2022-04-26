import numpy as np
import os , os.path as osp
import shutil
import json

data_name = 'hotdog'
raw_data_dir = '/home/wzpscott/NeuralRendering/shadownerf/data/V200L10'
data_dir = f'/home/wzpscott/NeuralRendering/shadownerf/data/nerf_synthetic_relight/{data_name}'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(osp.join(data_dir, 'image'), exist_ok=True)
os.makedirs(osp.join(data_dir, 'normal'), exist_ok=True)
os.makedirs(osp.join(data_dir, 'depth'), exist_ok=True)

for img_name in os.listdir(raw_data_dir):
    if img_name.endswith('.json'):
        continue # skip 'transforms.json'

    i = img_name.split('_')[0]
    raw_img_dir = osp.join(raw_data_dir, img_name)
    if 'depth' in img_name:
        img_dir = osp.join(data_dir, 'depth', f'{i}.png')
    elif 'normal' in img_name:
        img_dir = osp.join(data_dir, 'normal', f'{i}.png')
    else:  # rgb image
        img_dir = osp.join(data_dir, 'image', f'{i}')

    shutil.copy(raw_img_dir, img_dir)

with open(osp.join(raw_data_dir, 'transforms.json'), 'r') as fp:
    transforms = json.load(fp)
camera_angle_x = transforms['camera_angle_x']
frames = transforms['frames']

transforms_train, transforms_test = {}, {}
frames_train, frames_test = [], []
for i, frame in enumerate(frames):
    frame['file_path'] = osp.join(data_dir, 'image', f'{i}.png')
    if i<100:
        frames_train.append(frame)
    else:
        frames_test.append(frame)

transforms_train['camera_angle_x'] = camera_angle_x   
transforms_test['camera_angle_x'] = camera_angle_x  
transforms_train['frames'] = frames_train
transforms_test['frames'] = frames_test

with open(osp.join(data_dir,'transforms_train.json'), 'w') as fp:
    json.dump(transforms_train, fp, indent=4)
with open(osp.join(data_dir,'transforms_test.json'), 'w') as fp:
    json.dump(transforms_test, fp, indent=4)
