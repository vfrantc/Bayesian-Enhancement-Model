import os
import sys
class DummyFile(object):
    def write(self, x): pass
sys.stdout = DummyFile()
import argparse
from tqdm import tqdm 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import utils
from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
import lpips
from basicsr import calculate_niqe
sys.stdout = sys.__stdout__

parser = argparse.ArgumentParser(description='Image Enhancement')

parser.add_argument('--pred_dir', type=str, help='Dir of predited images')
parser.add_argument('--target_dir', type=str, default='', help='Dir of targets')
parser.add_argument('--psnr', action='store_true', help='True to compute PSNR')
parser.add_argument('--ssim', action='store_true', help='True to compute SSIM')
parser.add_argument('--lpips', action='store_true', help='True to compute LPIPS')
parser.add_argument('--niqe', action='store_true', help='True to compute NIQE')

args = parser.parse_args()

pred_paths = natsorted( glob(os.path.join(args.pred_dir, '*.png')) + glob(os.path.join(args.pred_dir, '*.jpg')) + glob(os.path.join(args.pred_dir, '*.bmp')) )
if args.target_dir != '':
    target_paths = natsorted( glob(os.path.join(args.target_dir, '*.png')) + glob(os.path.join(args.target_dir, '*.jpg')) + glob(os.path.join(args.target_dir, '*.bmp')) )

psnr = []
ssim = []
lpips_ = []
niqe = []

if args.lpips:
    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    loss_fn.cuda()

for p_idx, pred_path in tqdm(enumerate(pred_paths), total=len(pred_paths)):
    pred = np.float32(utils.load_img(pred_path)) / 255.
    if args.target_dir != '':
        target = np.float32(utils.load_img(target_paths[p_idx])) / 255.
    if args.psnr:
        psnr.append(utils.calculate_psnr(target, pred))
    if args.ssim:
        ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(pred)))
    if args.lpips:
        ex_p0 = lpips.im2tensor(img_as_ubyte(pred)).cuda()
        ex_ref = lpips.im2tensor(img_as_ubyte(target)).cuda()
        score_lpips = loss_fn.forward(ex_ref, ex_p0).item()
        lpips_.append(score_lpips)
    if args.niqe:
        niqe.append(calculate_niqe(img_as_ubyte(pred), crop_border=0))
if args.psnr:
    psnr = np.mean(np.array(psnr))
    print("Best_PSNR: {:.4f} dB".format(psnr))
if args.ssim:
    ssim = np.mean(np.array(ssim))
    print("Best_SSIM: {:.4f}".format(ssim))
if args.lpips:
    lpips_ = np.mean(np.array(lpips_))
    print("Best_lpips: {:.4f}".format(lpips_))
if args.niqe:
    niqe = np.mean(np.array(niqe))
    print("Best_NIQE: {:.4f}".format(niqe))


