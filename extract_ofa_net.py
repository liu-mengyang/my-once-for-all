import os
import torch
import random
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.model_zoo import ofa_net
from ofa.utils import MyModule
from ofa.logger import logger


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
parser.add_argument(
    "-n",
    "--net",
    metavar="OFANET",
    default="ofa_resnet50",
    choices=[
        "ofa_mbv3_d234_e346_k357_w1.0",
        "ofa_mbv3_d234_e346_k357_w1.2",
        "ofa_proxyless_d234_e346_k357_w1.3",
        "ofa_resnet50",
    ],
    help="OFA networks",
)
parser.add_argument("-ns", "--num_samples", default=1)

args = parser.parse_args()
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

ofa_network = ofa_net(args.net, pretrained=True)

def info_fw_hook(module, input, output):
    input_shape_list = []
    for i in input:
        input_shape_list.append(i.shape)
    logger.info(f"Information of {module} : {input_shape_list} {output.shape}")

def register_info_fw_hook(module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Sequential) or isinstance(sub_module, nn.ModuleList) or isinstance(sub_module, MyModule):
            register_info_fw_hook(sub_module)
        else:
            logger.info(f"Add hook to {sub_module}")
            sub_module.register_forward_hook(info_fw_hook)

for i in tqdm(range(int(args.num_samples))):
    """ Randomly sample a sub-network, 
        you can also manually set the sub-network using: 
            ofa_network.set_active_subnet(ks=7, e=6, d=4) 
    """
    logger.info("Sampling ...")
    ofa_network.sample_active_subnet()
    logger.info("Sampled")
    logger.info("Building ...")
    subnet = ofa_network.get_active_subnet(preserve_weight=True).cuda()
    logger.info("Built")
    
    """ Test sampled subnet 
    """
    # assign image size: 128, 132, ..., 224
    img_size = random.choice([128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224])
    
    # Output information
    logger.info("Registering ...")
    register_info_fw_hook(subnet)
    logger.info("Registered")
    logger.info("Forwarding ...")
    x = torch.randn(1, 3, img_size, img_size).cuda()
    y = subnet(x)
    logger.info("Forwarded")
    print(y.shape)