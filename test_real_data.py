from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

from lib import utils
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib
from lib.GaussianRender import pts2render
from torch.cuda.amp import autocast as autocast
from opt_flow.test import opt_flow
from lib.utils import flow2depth
import cuda_example
import matplotlib.pyplot as plt

from lib.loss import ssim, psnr

import torch
import warnings

import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import  structural_similarity as ssim
import time

warnings.filterwarnings("ignore", category=UserWarning)


class StereoHumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.dataset = StereoHumanDataset(self.cfg.dataset, phase=phase)
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()

    def read_image_rgba(self, filename):
        bgr_image = cv2.imread(filename)
        rgba = cv2.cvtColor(bgr_image ,cv2.COLOR_BGR2RGBA)

        return rgba

    def opt_flow(self, first_image, second_image):
        result = np.zeros((first_image.shape[0], first_image.shape[1], 2), dtype=np.uint16)
        cuda_example.opt_flow(first_image, second_image, result)
        
        # result is in S10.5
        result = result.astype(np.float32) / 32.0

        return result


    def infer_seqence(self, view_select, ratio=0.5):
        total_frames = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
        previous_frame_image_left = None
        previous_frame_image_right = None
        frame_depth_left = None
        frame_depth_right = None
        previous_gt = None
        psnr_values = [] 

        for idx in tqdm(range(total_frames)):
            item = self.dataset.get_test_item(idx, source_id=view_select)
            data = self.fetch_data(item)
            data = get_novel_calib(data, self.cfg.dataset, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')

            with torch.no_grad():
                bs = data['lmain']['img'].shape[0]
                image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
                with autocast(enabled=self.cfg.raft.mixed_precision):
                    img_feat = self.model.img_encoder(image)
                flow_up = self.model.raft_stereo(img_feat[2], iters=self.model.val_iters, test_mode=True)
                data['lmain']['flow_pred'] = flow_up[0]
                data['rmain']['flow_pred'] = flow_up[1]
                gt_frame_data = self.model.flow2gsparms(image, img_feat, data, bs)
                gt_data = pts2render(gt_frame_data, bg_color=self.cfg.dataset.bg_color)
                gt = self.tensor2np(gt_data['novel_view']['img_pred']) 
                gt_shifted = np.zeros_like(gt)
                gt_shifted[:, 1: 2048, :] = gt[:, :2047, :]

                print("shifted_psnr is", self.calculate_psnr(gt_shifted, gt))

                
                # if previous_gt is not None:
                #     fig, axes = plt.subplots(1, 3, figsize=(12, 5))
                #     axes[0].imshow(previous_gt)
                #     axes[0].set_title('previous_gt', fontsize=14, fontweight='bold')
                #     axes[0].axis('off')  # Hide axes
            
                #     # # Display second image
                #     axes[1].imshow(gt)
                #     axes[1].set_title('GT', fontsize=14, fontweight='bold')
                #     axes[1].axis('off')  # Hide axes

                #     axes[2].imshow(np.abs(gt - previous_gt))
                #     axes[2].set_title('diff', fontsize=14, fontweight='bold')
                #     axes[2].axis('off')  # Hide axes

                #     plt.show()
                #     print("gt comparison is", self.calculate_psnr(previous_gt, gt))
                    

                # previous_gt = gt

                if idx % 2 == 0:
                    frame_depth_left = gt_frame_data['lmain']['depth']
                    frame_depth_right = gt_frame_data['rmain']['depth']
                else:
                    start = time.time() 
                    left_opt_flow = opt_flow(data['lmain']['img_original'], previous_frame_image_left)
                    right_opt_flow = opt_flow(data['rmain']['img_original'], previous_frame_image_right)
                    finished_opt = time.time() - start
                    print("after opt time is", finished_opt)

                    left_opt_flow = torch.from_numpy(left_opt_flow).cuda() # [None, None, :, :].cuda()
                    right_opt_flow = torch.from_numpy(right_opt_flow).cuda() # [None, None, :, :].cuda()

                    finished_opt = time.time() - start
                    print("finished_np time is", finished_opt)
                    
                    y_grid_l, x_grid_l = torch.meshgrid(
                        torch.arange(1024, device=left_opt_flow.device, dtype=torch.float32).cuda(),
                        torch.arange(1024, device=left_opt_flow.device, dtype=torch.float32).cuda(),
                        indexing='ij'
                    )

                    x_grid_l = x_grid_l + left_opt_flow[:, :, 0]
                    y_grid_l = y_grid_l + left_opt_flow[:, :, 1]

                    x_normalized_l = 2.0 * x_grid_l / (1024 - 1) - 1.0
                    y_normalized_l = 2.0 * y_grid_l / (1024 - 1) - 1.0

                    grid_l = torch.stack([x_normalized_l, y_normalized_l], dim=-1).unsqueeze(0).type(torch.float32).cuda()
                    
                    warped_l = F.grid_sample(
                        frame_depth_left,
                        grid_l, 
                        mode='bilinear', 
                        padding_mode='zeros',
                    ).cuda()

                    finished_opt = time.time() - start
                    print("setup warpping", finished_opt)

                    y_grid_r, x_grid_r = torch.meshgrid(
                        torch.arange(1024, device=image.device, dtype=float).cuda(),
                        torch.arange(1024, device=image.device, dtype=float).cuda(),
                        indexing='ij'
                    )

                    x_grid_r = x_grid_r + right_opt_flow[:, :, 0]
                    y_grid_r = y_grid_r + right_opt_flow[:, :, 1]
                    x_normalized_r = 2.0 * x_grid_r / (1024 - 1) - 1.0
                    y_normalized_r = 2.0 * y_grid_r / (1024 - 1) - 1.0

                    grid_r = torch.stack([x_normalized_r, y_normalized_r], dim=-1).unsqueeze(0).type(torch.float32).cuda()
                    
                    warped_r = F.grid_sample(
                        frame_depth_right,
                        grid_r, 
                        mode='bilinear', 
                        padding_mode='zeros',
                        align_corners=True
                    )
                    finished_opt = time.time() - start
                    print("finished_warping", finished_opt)

                    frame_data = self.model.flow2gsparms(
                        image, 
                        img_feat, 
                        data, bs, 
                        override_depth = {
                            'lmain': warped_l,
                            'rmain': warped_r, 
                        }
                    )
                    data = pts2render(frame_data, bg_color=self.cfg.dataset.bg_color)
                    
                previous_frame_image_left = data['lmain']['img_original']
                previous_frame_image_right = data['rmain']['img_original']

                render_novel = self.tensor2np(data['novel_view']['img_pred'])
                if idx % 2 != 0:
                    psnr_val = ssim(render_novel, gt)
                    print("PSNR is", psnr_val)
                    psnr_values.append(psnr_val)
                    # fig, axes = plt.subplots(1, 3, figsize=(12, 5))
                    # axes[0].imshow(render_novel)
                    # axes[0].set_title('First Image', fontsize=14, fontweight='bold')
                    # axes[0].axis('off')  # Hide axes
            
                    # # # Display second image
                    # axes[1].imshow(gt)
                    # axes[1].set_title('GT', fontsize=14, fontweight='bold')
                    # axes[1].axis('off')  # Hide axes

        counts, bin_edges = np.histogram(np.array(psnr_values), bins=50)
        cdf = np.cumsum(counts) / np.sum(counts)
        plt.figure(figsize=(12, 5))
        
        # Plot 1: CDF from histogram
        plt.subplot(1, 2, 1)
        plt.plot(bin_edges[1:], cdf, linewidth=2, color='blue')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF using np.histogram')
        plt.grid(True, alpha=0.3)
        print("average PSNR is", sum(psnr_values)/len(psnr_values))
        plt.show()



    def calculate_psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = max(np.max(img1), np.max(img2))
        print("max pixel is", max_pixel)
        import math
        psnr = 10 * math.log10(max_pixel**2 / mse)
        return psnr
    

    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                if item == 'img_original':
                    continue
                data[view][item] = data[view][item].cuda().unsqueeze(0)
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_view', type=int, nargs='+', required=True)
    parser.add_argument('--ratio', type=float, default=0.5)
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage2.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = arg.ckpt_path
    cfg.test_out_path = './test_out'
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = StereoHumanRender(cfg, phase='test')
    render.infer_seqence(view_select=arg.src_view, ratio=arg.ratio)
