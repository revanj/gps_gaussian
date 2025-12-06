from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

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

import torch
import warnings
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
        frame_data = None
        frame_depth_left = None
        frame_depth_right = None

        for idx in tqdm(range(total_frames)):
            item = self.dataset.get_test_item(idx, source_id=view_select)
            data = self.fetch_data(item)
            data = get_novel_calib(data, self.cfg.dataset, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')
            with torch.no_grad():
                if idx % 2 == 0:
                    print("generating depth")
                    bs = data['lmain']['img'].shape[0]
                    image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
                    with autocast(enabled=self.cfg.raft.mixed_precision):
                        img_feat = self.model.img_encoder(image)
                    flow_up = self.model.raft_stereo(img_feat[2], iters=self.model.val_iters, test_mode=True)
                    print("finished raft stereo")
                    data['lmain']['flow_pred'] = flow_up[0]
                    data['rmain']['flow_pred'] = flow_up[1]
                    frame_data = self.model.flow2gsparms(image, img_feat, data, bs)
                    frame_depth_left = frame_data['lmain']['depth']
                    frame_depth_right = frame_data['rmain']['depth']
                    print("finished flow2gsparams")
                else:
                    # mutate frame_data via cuda
                    left_opt_flow = opt_flow(previous_frame_image_left, data['lmain']['img_original'])
                    right_opt_flow = opt_flow(previous_frame_image_right, data['rmain']['img_original'])

                    left_opt_flow = torch.from_numpy(left_opt_flow) # [None, None, :, :].cuda()
                    right_opt_flow = torch.from_numpy(right_opt_flow) # [None, None, :, :].cuda()
                    
                    new_depth_l = torch.zeros_like(frame_depth_right)
                    new_depth_r = torch.zeros_like(frame_depth_right)

                    # for i in range(1024):
                    #     for j in range(1024):
                    #         left_flow_x = int(left_opt_flow[i, j, 0])
                    #         left_flow_y = int(left_opt_flow[i, j, 1])

                    #         right_flow_x = int(right_opt_flow[i, j, 0])
                    #         right_flow_y = int(right_opt_flow[i, j, 1])

                    #         if 0 <= (i + left_flow_x) and (i + left_flow_x) < 1024 and 0 <= j + left_flow_y and j + left_flow_y < 1024:
                    #             new_depth_l[i, j, :] = frame_depth_left[i + left_flow_x, j + left_flow_y]



                    frame_data = self.model.flow2gsparms(
                        image, 
                        img_feat, 
                        data, bs, 
                        override_depth = {
                            'lmain': left_opt_flow, 
                            'rmain': right_opt_flow
                        }
                    )

                
                print("starting pts render")
                # data, _, _ = self.model(data, is_train=False)
                data = pts2render(frame_data, bg_color=self.cfg.dataset.bg_color)
                previous_frame_image_left = data['lmain']['img_original'];
                previous_frame_image_right = data['rmain']['img_original'];


            render_novel = self.tensor2np(data['novel_view']['img_pred'])
            cv2.imwrite(self.cfg.test_out_path + '/%s_novel.jpg' % (data['name']), render_novel)

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
