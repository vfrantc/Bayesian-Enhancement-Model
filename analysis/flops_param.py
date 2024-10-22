import basicsr.archs as archs
# from model_zoo.HWMNet import buildHWMNet
# from basicsr.archs.RetinexMamba_arch import RetinexMamba
# from model_zoo.RetinexFormer import buildRetinexFormer
# from model_zoo.UVMNet import buildUVMNet
# from model_zoo.LLFormer import buildLLFormer
from analysis.util import FLOPs
from analysis.util import Throughput
from basicsr.bayesian import convert2bnn, convert2bnn_selective
import logging
import torch
from torch.utils.data import Dataset, DataLoader

class RandomTensorDataset(Dataset):
    def __init__(self, size=(3, 256, 256), length=30):
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(self.size).cuda(), 0

if __name__ == '__main__':

    H, W = 16, 16
    model = archs.BUNet_arch.build_model()
    bnn_config = {
            "sigma_init": 0.05,
            "decay": 0.998,
            "pretrain": False,
    }
    convert2bnn_selective(model, bnn_config)
    # model = archs.UNet_arch.build_model()
    # model = buildHWMNet()
    # model = RetinexMamba(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2])
    # model = buildRetinexFormer()
    # model = buildUVMNet()
    # model = buildLLFormer()

    model.cuda()

    n_param, flops = FLOPs.fvcore_flop_count(model, input_shape=(1, 3, H, W), verbose=False)
    print(f'FLOPs:{flops:.3f}G')
    print(f'Params:{n_param/(1000*1000):.3f}M')

    dataset = RandomTensorDataset(size=(3, H, W), length=30)
    dataloader = DataLoader(dataset, batch_size=1)
    Throughput.throughput(data_loader=dataloader, model=model, logger=logging)


