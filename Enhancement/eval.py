import os
import random
import time
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
import utils
from natsort import natsorted
from glob import glob
from PIL import Image
from skimage import img_as_ubyte
import lpips
from basicsr.models import build_model
from basicsr.utils.options import parse
from basicsr.bayesian import set_prediction_type
from basicsr.utils.histogram import compute_histograms
from basicsr.metrics import calculate_niqe
from basicsr.metrics import getUCIQE, getUIQM
from torchmetrics.multimodal import CLIPImageQualityAssessment




parser = argparse.ArgumentParser(description='Image Enhancement')

parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--input_dir', default='', type=str, help='Directory for inputs')
parser.add_argument('--target_dir', default='', type=str, help='Directory for targets')
parser.add_argument('--opt', type=str, default='youryaml.yaml', help='Path to option YAML file.')
parser.add_argument('--cond_opt', type=str, default='your condition_yaml.yaml', help='Path to option YAML file.')
parser.add_argument('--weights', default='yourweight.pth', type=str, help='Path to weights')
parser.add_argument('--cond_weights', default='yourweight.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='yourdataset', type=str, help='Name of dataset')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--num_samples', default=200, type=int, help='Number of random samples')
parser.add_argument('--Monte_Carlo', action='store_true', help='use Monte Carlo Simulation, i.e., averaging the outcome of random samples. \
                    when the smaple number is very large, Monte Carlo Simulation is equal to the deterministic model')
parser.add_argument('--psnr_weight', default=1.0, type=float, help='Balance between PSNR and SSIM')
parser.add_argument('--no_ref', default='', type=str, choices=['clip', 'niqe', 'uiqm_uciqe'], help='no reference image quality evaluator. \
                    support CLIP-IQA and NIQE')
parser.add_argument('--uiqm_weight', default=1.0, type=float, help='Balance between UIQM and UICIQE')
parser.add_argument('--lpips', action='store_true', help='True to compute LPIPS')
parser.add_argument('--deterministic', action='store_true', help='Use deterministic mode')
parser.add_argument('--seed', default=287128, type=int, help='fix random seed to reproduce consistent resutls')

args = parser.parse_args()

#-------------------Set random seed----------------
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#-------------------Load yaml----------------------
yaml_file = args.opt
weights = args.weights
cond_weights = args.cond_weights
print(f"dataset {args.dataset}")

opt = parse(args.opt, is_train=False)
opt['dist'] = False
cond_opt = parse(args.cond_opt, is_train=False)
cond_opt['dist'] = False

net = build_model(opt).net_g
set_prediction_type(net, deterministic=args.deterministic)
cond_net = build_model(cond_opt).net_g

checkpoint = torch.load(weights)
cond_checkpoint = torch.load(cond_weights)

scale_factor = opt['condition'].get('scale_down', 0) + opt['condition'].get('hist_patch_size', 0)

if args.deterministic:
    args.num_samples = 1

net.load_state_dict(checkpoint['params'])
print("Loaded weights from", weights)
cond_net.load_state_dict(cond_checkpoint['params'])
print("Loaded weights from", cond_weights)


net.cuda()
net = nn.DataParallel(net)
net.eval()

cond_net.cuda()
cond_net = nn.DataParallel(cond_net)
cond_net.eval()

factor = 32
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
os.makedirs(result_dir, exist_ok=True)


if args.no_ref =='clip':
    # clip_metric = CLIPImageQualityAssessment(prompts=("noisiness",)).cuda()
    # clip_metric = CLIPImageQualityAssessment(prompts=("quality","brightness", "natural", 'noisiness', 'colorfullness', 'sharpness', 'contrast', 'beautiful')).cuda()
    # clip_metric = CLIPImageQualityAssessment(prompts=("quality", "natural", 'noisiness')).cuda()
    # clip_metric = CLIPImageQualityAssessment(prompts=('warm',)).cuda()
    clip_metric = CLIPImageQualityAssessment(prompts=("brightness", "natural", 'noisiness')).cuda()
psnr = []
ssim = []
lpips_ = []
niqe = []
uiqm = []
uciqe = []
if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    raise ValueError('Not implemented yet')

if args.input_dir != '':
    input_dir = args.input_dir
    target_dir = args.target_dir
else:
    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
if args.GT_mean and target_dir =='':
    raise ValueError('GT_mean is only available when GT is provided')
input_paths = natsorted( glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.bmp')) )
if target_dir != '':
    target_paths = natsorted( glob(os.path.join(target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')) + glob(os.path.join(target_dir, '*.bmp')) )

if args.lpips:
    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    loss_fn.cuda()
def _padimg_np(inp, factor=factor):
    h, w = inp.shape[0], inp.shape[1]
    hp, wp = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = hp - h if h % factor != 0 else 0
    padw = wp - w if w % factor != 0 else 0
    if padh != 0 or padw !=0:
        inp = np.pad(inp, ((0, padh), (0, padw), (0, 0)), 'reflect')
    return inp

if len(input_paths) == 0:
    raise ValueError('No input images found')
mc_psnr = []
mc_ssim = []
start_time = time.perf_counter()
with torch.inference_mode():
    for p_idx, inp_path in tqdm(enumerate(input_paths), total=len(input_paths)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path)) / 255.
        h, w = img.shape[0], img.shape[1]
        if target_dir != '':
            target = np.float32(utils.load_img(target_paths[p_idx])) / 255.
            target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).cuda()
        img_pad = _padimg_np(img, factor=4*scale_factor) # Paddings
        hp, wp = img_pad.shape[0], img_pad.shape[1]
        input_ = torch.from_numpy(img_pad).permute(2, 0, 1).unsqueeze(0).cuda()

        if opt['condition']['type'] == 'mean':
            img_down = cv2.resize(img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LINEAR)
            dh, dw = img_down.shape[0], img_down.shape[1]
            img_down_pad = _padimg_np(img_down, factor=4) # Paddings
            img_down_pad = torch.from_numpy(img_down_pad).permute(2, 0, 1).unsqueeze(0).cuda()
        elif opt['condition']['type'] == 'histogram':
            hist_lq = compute_histograms(img, opt['condition']['hist_patch_size'], opt['condition']['num_bins'])
            hist_lq = hist_lq.permute(1 ,2, 3, 0)
            hist_lq = hist_lq.reshape(hist_lq.shape[0], hist_lq.shape[1], -1).numpy()
            dh, dw = hist_lq.shape[0], hist_lq.shape[1]
            hist_lq_pad = _padimg_np(hist_lq)
            hist_lq_pad = torch.from_numpy(hist_lq_pad).permute(2, 0, 1).unsqueeze(0).cuda()

        one_pred_list = []
        one_psnr_list = []
        one_ssim_list = []
        one_niqe_list = []
        one_clip_list =[]
        one_uiqm_list = []
        one_uciqe_list = []
        one_pred_cond_list = []
        if args.Monte_Carlo:
            mc_pred_list = []

        if opt['condition']['type'] == 'mean':
            input_cond = img_down_pad
        elif opt['condition']['type'] == 'histogram':
            input_cond = hist_lq_pad
        for i in range(args.num_samples):
            pred_cond = net(input_cond)[-1]
            pred_cond = torch.clamp(pred_cond, 0, 1)
            if args.GT_mean and opt['condition']['type'] != 'histogram':
                mean_pred = pred_cond.mean(dim=(2,3), keepdims=True)
                mean_target = target_tensor.mean(dim=(2,3), keepdims=True)
                pred_cond = torch.clamp(pred_cond * (mean_target / mean_pred), 0, 1)

            # if opt['condition']['type'] == 'histogram':
            noise_level = cond_opt['condition'].get('noise_level', 0)
            pred_cond = pred_cond + torch.randn_like(pred_cond) * noise_level

            pred_cond = F.interpolate(pred_cond, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            one_pred_cond_list.append(pred_cond)

        one_pred_conds = torch.cat(one_pred_cond_list, dim=0)
        one_pred_conds = one_pred_conds[:, :, :hp, :wp]

        input_expended = input_.tile(args.num_samples, 1, 1, 1)
        cat_input = torch.cat([input_expended, one_pred_conds], dim=1)
        split_input = cat_input.split(8)

        one_preds = []
        for split in split_input:
            one_preds.append(cond_net(split)[-1])
        one_preds = torch.cat(one_preds, dim=0)
        one_preds = one_preds[:, :, :h, :w]

        for i in range(args.num_samples):
            pred = one_preds[i:i+1]
            if args.Monte_Carlo:
                mc_pred_list.append(pred.cpu().detach().permute(0, 2, 3, 1).squeeze(0))

            pred_clip = pred
            pred = torch.clamp(pred, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                mean_pred = pred.mean(axis=(0,1), keepdims=True)
                mean_target = target.mean(axis=(0,1), keepdims=True)
                pred = np.clip(pred * (mean_target / mean_pred), 0, 1)

            one_pred_list.append(pred)
            if args.no_ref == 'clip':
                vs = clip_metric(pred_clip)
                vs = [vs] if isinstance(vs, torch.Tensor) else vs.values()
                one_clip_list.append(np.mean([v.item() for v in vs]))
            elif args.no_ref == 'niqe':
                one_niqe_list.append(calculate_niqe(pred*255, crop_border=0))
            elif args.no_ref == 'uiqm_uciqe':
                img_RGB = np.array(
                    Image.fromarray(img_as_ubyte(pred)).resize((256, int(256 / pred.shape[1] * pred.shape[0])))
                )
                one_uiqm_list.append(getUIQM(img_RGB))
                one_uciqe_list.append(getUCIQE(img_as_ubyte(pred)))
            else:
                one_psnr_list.append(utils.calculate_psnr(target, pred))
                one_ssim_list.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(pred)))

        #------------------------------------------------------------------------------------------

        if args.no_ref in ['clip', 'niqe', 'uiqm_uciqe']:
            if args.no_ref == 'clip':
                _idx = one_clip_list.index(max(one_clip_list))
            elif args.no_ref == 'niqe':
                _v = min(one_niqe_list)
                _idx = one_niqe_list.index(_v)
                niqe.append(_v)
            elif args.no_ref == 'uiqm_uciqe':
                best_one_list = (args.uiqm_weight * np.array(one_uiqm_list) / max(one_uiqm_list) + (1-args.uiqm_weight)* np.array(one_uciqe_list) / max(one_uciqe_list)).tolist()
                _idx = best_one_list.index(max(best_one_list))
                uiqm.append(one_uiqm_list[_idx])
                uciqe.append(one_uciqe_list[_idx])
            best_one_pred = one_pred_list[_idx]
            if target_dir != '':
                psnr.append(utils.calculate_psnr(target, best_one_pred))
                ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(best_one_pred)))
        else:
            best_one_list = (args.psnr_weight * np.array(one_psnr_list) / max(one_psnr_list)  + (1 - args.psnr_weight) * np.array(one_ssim_list) / max(one_ssim_list)).tolist()
            _idx = best_one_list.index(max(best_one_list))
            best_one_pred = one_pred_list[_idx]
            best_one_psnr = one_psnr_list[_idx]
            best_one_ssim = one_ssim_list[_idx]
            psnr.append(best_one_psnr)
            ssim.append(best_one_ssim)

        if target_dir != '':
            if args.lpips:
                ex_p0 = lpips.im2tensor(img_as_ubyte(best_one_pred)).cuda()
                ex_ref = lpips.im2tensor(img_as_ubyte(target)).cuda()
                score_lpips = loss_fn.forward(ex_ref, ex_p0).item()
                lpips_.append(score_lpips)

        if args.Monte_Carlo:
            mc_pred = torch.clamp(torch.mean(torch.stack(mc_pred_list), dim=0), 0, 1).numpy()
            if args.GT_mean:
                mean_mc_pred = cv2.cvtColor(mc_pred.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mc_pred = np.clip(mc_pred * (mean_target / mean_mc_pred), 0, 1)
            if target_dir != '':
                mc_psnr.append(utils.calculate_psnr(target, mc_pred))
                mc_ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(mc_pred)))

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(best_one_pred))

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"running time: {execution_time:.4f} sec")

with open(os.path.join(result_dir, 'result.txt'), 'w') as f:
    if target_dir != '':
        psnr = np.mean(np.array(psnr))
        ssim = np.mean(np.array(ssim))
        print("Best_PSNR: {:.4f} dB".format(psnr))
        print("Best_SSIM: {:.4f}".format(ssim))
        f.write("Best_PSNR: {:.4f} dB \n".format(psnr))
        f.write("Best_SSIM: {:.4f} \n".format(ssim))
        if args.lpips:
            lpips_ = np.mean(np.array(lpips_))
            print("Best_lpips: {:.4f}".format(lpips_))
            f.write("Best_lpips: {:.4f} \n".format(lpips_))

    if args.no_ref == 'niqe':
        niqe = np.mean(np.array(niqe))
        print("Best_NIQE: {:.4f}".format(niqe))
        f.write("Best_NIQE: {:.4f} \n".format(niqe))

    if args.no_ref == 'uiqm_uciqe':
        uiqm = np.mean(np.array(uiqm))
        uciqe = np.mean(np.array(uciqe))
        print("Best_UIQM: {:.4f}".format(uiqm))
        print("Best_UCIQE: {:.4f}".format(uciqe))
        f.write("Best_UIQM: {:.4f} \n".format(uiqm))
        f.write("Best_UCIQE: {:.4f} \n".format(uciqe))

    if args.Monte_Carlo and target_dir != '':
        mc_psnr = np.mean(np.array(mc_psnr))
        mc_ssim = np.mean(np.array(mc_ssim))
        print("MC_PSNR: {:.4f} dB".format(mc_psnr))
        print("MC_SSIM: {:.4f}".format(mc_ssim))
        f.write("MC_PSNR: {:.4f} dB \n".format(mc_psnr))
        f.write("MC_SSIM: {:.4f} \n".format(mc_ssim))