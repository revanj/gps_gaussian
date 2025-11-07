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

import torch
import warnings

from PIL import Image
import torchvision.transforms as T

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

    def run_inference(self, view_select, idx, ratio=0.5, ldepth = None, rdepth = None):
        item = self.dataset.get_test_item(idx, source_id=view_select)
        data = self.fetch_data(item)
        data = get_novel_calib(data, self.cfg.dataset, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')
        with torch.no_grad():
            data, _, _ = self.model(data, is_train=False, ldepth = ldepth, rdepth = rdepth)
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)
        return data

    def psnr(self, img1, img2):
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def infer_seqence(self, view_select, ratio=0.5):
        total_psnr = 0
        counter = 0
        total_frames = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
        for idx in tqdm(range(2, total_frames)):
            depth_l = Image.open("/home/revanj/repos/avatar/gps_gaussian/depth_interpolate/left_depth/{:04d}_left_depth.jpg".format(idx-1)).convert("L")
            depth_r = Image.open("/home/revanj/repos/avatar/gps_gaussian/depth_interpolate/right_depth/{:04d}_right_depth.jpg".format(idx-1)).convert("L")
            flow_l = np.load("/home/revanj/repos/avatar/gps_gaussian/flow_bg_npy/left_view/{:04d}_flow.jpg.npy".format(idx)).astype(np.float32)
            flow_l = np.floor(flow_l).astype(np.uint32)
            flow_r = np.load("/home/revanj/repos/avatar/gps_gaussian/flow_bg_npy/right_view/{:04d}_flow.jpg.npy".format(idx)).astype(np.float32)
            flow_r = np.floor(flow_r).astype(np.uint32)
            to_tensor = T.ToTensor()
            depth_l = to_tensor(depth_l).unsqueeze(0).cuda()
            depth_r = to_tensor(depth_r).unsqueeze(0).cuda()
            depth_l_new = depth_l.clone()
            depth_r_new = depth_r.clone()

            for (flow, depth, depth_new) in [(flow_l, depth_l, depth_l_new), (flow_r, depth_r, depth_r_new)]:
                for i in range(0, 1024):
                    for j in range(0, 1024):
                        target_h = i + flow[i, j, 0]
                        target_w = j + flow[i, j, 1]
                        if 0 <= target_h < 1024 and 0 <= target_w < 1024:
                            depth_new[0, 0, target_h, target_w] = depth[0, 0, i, j]

            # data = self.run_inference(view_select, idx, ratio=ratio, ldepth=depth_l, rdepth=depth_r)
            data = self.run_inference(view_select, idx-2, ratio=ratio, ldepth=depth_l_new, rdepth=depth_r_new)
            # data = self.run_inference(view_select, idx-2, ratio=ratio)
            # data = self.run_inference(view_select, idx-2, ratio=ratio)
            # render_novel = self.tensor2np(data['novel_view']['img_pred'])
            # ldepth =  self.tensor2np(data['lmain']['depth'])
            # rdepth = self.tensor2np(data['rmain']['depth'])
            # lview = self.tensor2np_img(data['lmain']['img'])
            # rview = self.tensor2np_img(data['rmain']['img'])

            render_novel = self.tensor2np(data['novel_view']['img_pred'])
            # gt_novel = data['novel_view']['img_pred']
            # psnr_value = self.psnr(render_novel, gt_novel).mean().double()
            # total_psnr += psnr_value
            # counter += 1
            # print("psnr is", psnr_value)

            # if idx % 2 == 0:
            cv2.imwrite(self.cfg.test_out_path + '/%s_novel.jpg' % (data['name']), render_novel)
            # cv2.imwrite(self.cfg.test_out_path + '/%s_left_view_bg.jpg' % (data['name']), lview)
            # cv2.imwrite(self.cfg.test_out_path + '/%s_right_view_bg.jpg' % (data['name']), rview)

        print("average psnr is", total_psnr/counter)


    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        # img_np = img_tensor[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def tensor2np_img(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        # img_np = img_tensor[0].detach().cpu().numpy()
        img_np = (img_np/2.0 + 0.5) * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
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
