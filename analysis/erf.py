import os
import time
from functools import partial
from typing import Callable
import seaborn
# from model_zoo.swinIR import buildSwinIR
# from model_zoo.rcan import buildRCAN
# from model_zoo.edsr import buildEDSR
# from model_zoo.hat import HAT
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
# from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
# from basicsr.data.transforms import augment, paired_random_crop
# from basicsr.utils import FileClient, imfrombytes, img2tensor
# from basicsr.utils.matlab_functions import rgb2ycbcr
import torch
import torch.nn as nn
from torch import optim as optim
from torchvision import datasets, transforms
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from basicsr.utils.options import dict2str, parse_options
from basicsr.data.paired_image_dataset import Dataset_PairedImage
root_path = '/home/edward/UIEMamba/Options/SuperMamba161_222_LOLv1.yml'
opt, _ = parse_options(root_path, is_train=False)
opt=opt['datasets']['val'] # we use the 4-th SR testsets(i.e. Urban100) to visualize ERF.



if True:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns

    #   Set figure parameters
    large = 24;
    med = 24;
    small = 24
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white")
    # plt.rc('font', **{'family': 'Times New Roman'})
    plt.rcParams['axes.unicode_minus'] = False



# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def analyze_erf(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25)):
    def heatmap(data, camp='RdYlGn', figsize=(10, 10), ax=None, save_path=None):
        plt.figure(figsize=figsize, dpi=40)
        ax = sns.heatmap(data,
                         xticklabels=False,
                         yticklabels=False, cmap=camp,
                         center=0, annot=False, ax=ax, cbar=False, annot_kws={"size": 24}, fmt='.2f')
        plt.savefig(save_path)

    def analyze_erf(args):
        data = args.source
        print(np.max(data))
        print(np.min(data))
        data = args.ALGRITHOM(data + 1)  # the scores differ in magnitude. take the logarithm for better readability
        data = data / np.max(data)  # rescale to [0,1] for the comparability among models
        heatmap(data, save_path=args.heatmap_save)
        print('heatmap saved at ', args.heatmap_save)

    class Args():
        ...

    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM
    os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)
    analyze_erf(args)


# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def visualize_erf(MODEL: nn.Module = None, num_images=100, 
                  save_path=f"/tmp/{time.time()}/erf.npy"):
    def get_input_grad(model, samples):
        outputs = model(samples)
        out_size = outputs.size()
        central_point = outputs[:, :, out_size[2] // 2, out_size[3] // 2].sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    def main(args, MODEL: nn.Module = None):
        dataset = Dataset_PairedImage(opt)
        test_loader = data.DataLoader(dataset,batch_size=1,shuffle=False)

        model = MODEL
        model.cuda().eval()

        optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

        meter = AverageMeter()
        optimizer.zero_grad()

        for idx,data_sample in enumerate(test_loader):
            if meter.count == args.num_images:
                return meter.avg
            # we set the imhg size to 120X120 due to the GPU memory constrain
            samples = F.interpolate(data_sample['lq'],size=(120,120))
            samples = samples.cuda(non_blocking=True)
            samples.requires_grad = True
            optimizer.zero_grad()
            contribution_scores = get_input_grad(model, samples)
            torch.cuda.empty_cache()
            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                print(f'accumulat{idx}')
                meter.update(contribution_scores)

        return meter.avg


    class Args():
        ...

    args = Args()
    args.num_images = num_images
    args.save_path = save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return main(args, MODEL)




if __name__ == '__main__':
    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "show/erf")
    kwargs = dict(only_backbone=True, with_norm=False)
    from basicsr.models.archs.SuperMamba_arch import SuperMamba
    init_model = SuperMamba(stage=1,n_feat=40,num_blocks=[2,2,2], d_state=16, ssm_ratio=1)
    # ckpt_path = '/home/edward/UIEMamba/experiments/SuperMamba161_222_LOLv1/models/net_g_1000.pth' # path to load your pre_trained model weights
    # init_model.load_state_dict(torch.load(ckpt_path)['params'])
    save_path = f"./tmp/{time.time()}/erf.npy"
    grad_map = visualize_erf(init_model, save_path=save_path)
    analyze_erf(source=grad_map, dest=f"{showpath}/erf.png")

