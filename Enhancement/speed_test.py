import random
import argparse
import numpy as np
import time
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models import build_model
from basicsr.utils.options import parse
from basicsr.bayesian import set_prediction_type

parser = argparse.ArgumentParser(description='Image Enhancement')


parser.add_argument('--opt', type=str, default='youryaml.yaml', help='Path to option YAML file.')
parser.add_argument('--cond_opt', type=str, default='your condition_yaml.yaml', help='Path to option YAML file.')
parser.add_argument('--weights', default='yourweight.pth', type=str, help='Path to weights')
parser.add_argument('--cond_weights', default='yourweight.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='yourdataset', type=str, help='Name of dataset')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--num_smaples', default=5, type=int, help='Number of random samples')
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



scale_factor = 16
input_size = 2048
speedup = False
repeat = 100
start_time = time.perf_counter()
with torch.inference_mode():

    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    input = torch.randn(1, 3, input_size, input_size).cuda()

    if speedup:
        # mocking parallel computation
        tiled_input  = input.tile(repeat, 1, 1, 1)
        tiled_input = F.interpolate(tiled_input, size=(input_size//scale_factor, input_size//scale_factor), mode='bilinear', align_corners=False)
        pred_cond = net(tiled_input)[-1]

        # mocking the computation time for metric D
        F.mse_loss(pred_cond, pred_cond, reduction='mean')

        # mocking choosing the best prediction
        pred_cond = pred_cond[0:1]

        pred_cond = F.interpolate(pred_cond, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        pred = cond_net(torch.cat([input, pred_cond], dim=1))[-1]
    else:
        for i in range(repeat):
            pred_cond = net(input)[-1]
            pred = cond_net(torch.cat([input, pred_cond], dim=1))[-1]


end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"running time: {execution_time:.4f} sec")