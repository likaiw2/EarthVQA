import os
import sys

import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import numpy as np
import os
from tqdm import tqdm
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
import argparse
import h5py
from PIL import Image
import torchvision.transforms as T
import albumentations as A
from skimage.io import imread
from albumentations.pytorch import ToTensorV2

# 定义颜色映射（0~9）
COLOR_MAP = {
    0: (0, 0, 0),              # nothing
    1: (255, 255, 255),        # Background
    2: (255, 0, 0),            # Building
    3: (255, 255, 0),          # Road
    4: (0, 0, 255),            # Water
    5: (159, 129, 183),        # Barren
    6: (0, 255, 0),            # Forest
    7: (255, 195, 128),        # Agricultural
    8: (165, 0, 165),          # Playground
    9: (0, 185, 246),          # Pond
}


logger = logging.getLogger(__name__)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
er.registry.register_all()

def predict_seg(ckpt_path, config_path, save_dir, image_path):
    cfg = import_config(config_path)
    #model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)
    model_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    log_dir = os.path.dirname(ckpt_path)
    # test_dataloader = make_dataloader(cfg['data']['test'])
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    image = imread(image_path)
    
    transform = A.Compose([
        A.Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, 
                  always_apply=True),
        ToTensorV2()
    ])

    blob = transform(image=image)
    image = blob['image']
    
    img_tensor = image.unsqueeze(0).cuda()
    # print(img_tensor.shape)

    with torch.no_grad():
        pred, img_feat = model(img_tensor)
        # pred = pred.argmax(dim=1).cpu().squeeze(0).numpy().astype(np.uint8)
        imname = os.path.basename(image_path)
        
        # print(pred.shape)
        pred = pred[0]
        img_feat = img_feat[0]

        # Save the prediction and features
        pred = pred.argmax(dim=0).cpu()
        clsmap = pred.numpy().astype(np.uint8)
        hdf_path = os.path.join(save_dir, imname.replace('.png', '.hdf5'))
        f = h5py.File(hdf_path, 'w')
        f.create_dataset('feature', data=img_feat.cpu().numpy())
        f.create_dataset('pred_mask', data=clsmap+1)
        f.close()
        
        # 生成彩色可视化图
        pred_mask = clsmap+1
        # print(pred_mask.shape)
        # print(np.unique(pred_mask))
        
        h, w = pred_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in COLOR_MAP.items():
            color_mask[pred_mask == label] = color
        vis_img = Image.fromarray(color_mask)
        vis_path = hdf_path.replace('.hdf5', '_pred.png')
        # print(vis_path)
        vis_img.save(vis_path)

        torch.cuda.empty_cache()
        return vis_img

if __name__ == '__main__':
    # orig_cwd = os.getcwd()
    # er.registry.register_all()
    parser = argparse.ArgumentParser(description='Eval methods')
    parser.add_argument('--ckpt_path',  type=str,
                        help='ckpt path', 
                        # default='./log/deeplabv3p.pth',
                        default='weights/sfpnr50.pth'
                        )
    parser.add_argument('--config_path',  type=str,
                        help='config path', 
                        # default='sfpnr50',
                        default='configs/sfpnr50.py',
                        )
    parser.add_argument('--save_dir',  type=str,
                        help='save dir', 
                        default='streamlit_script/out'
                        )
    parser.add_argument('--image_path', 
                        type=str, 
                        help='Path to input image',
                        default='dataset/EarthVQA/Test/4193.png'
                        )

    args = parser.parse_args()
    predict_seg(args.ckpt_path, args.config_path, args.save_dir,image_path=args.image_path)
    # os.chdir(orig_cwd)