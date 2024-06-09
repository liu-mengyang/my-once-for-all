import argparse
import json
import os
import random
import sys
import time
import torch

from tqdm import tqdm

import torch
import torch.nn as nn

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.model_zoo import ofa_net
from ofa.utils import MyModule
from ofa.logger import logger

from autotailor.ruler.profiler.profiler import Profiler
from infra.convertor.convertor import Convertor


parser = argparse.ArgumentParser()
parser.add_argument("backend_config_path", type=str, default=None)
parser.add_argument("command_config_path", type=str, default=None)
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default=None)
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
parser.add_argument("--add_register", action="store_true", default=False)
parser.add_argument("-ns", "--num_samples", default=None)
parser.add_argument("-log", "--log_file", default=None)

args = parser.parse_args()

if args.log_file:
    logger.add(args.log_file)

if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
elif args.gpu:
    device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# load connector and convertor
backend_config = json.load(open(args.backend_config_path))
command_config = json.load(open(args.command_config_path))

profiler = Profiler(backend_config, command_config)

convertor = Convertor()

ofa_network = ofa_net(args.net, pretrained=True)

if args.num_samples:
    convert_cost = 0
    profile_cost = 0
    total_blocks = 0
    for i in tqdm(range(int(args.num_samples))):
        """ Randomly sample a sub-network, 
            you can also manually set the sub-network using: 
                ofa_network.set_active_subnet(ks=7, e=6, d=4) 
        """
        logger.info("Sampling ...")
        ofa_network.sample_active_subnet()
        logger.info("Sampled")
        logger.info("Building ...")
        subnet = ofa_network.get_active_subnet(preserve_weight=True)
        if args.gpu:
            subnet.cuda()
        logger.info("Built")
        
        """ Test sampled subnet 
        """
        # assign image size: 128, 132, ..., 224
        img_size = random.choice([128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224])
        if args.net == "ofa_mbv3_d234_e346_k357_w1.2":
            img_size = random.choice([160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224])
        
        x = torch.randn(1, 3, img_size, img_size)
        if args.gpu:
            x = x.cuda()
            
        y, num_blocks, convert_time, profile_time = subnet.forward_for_export_and_profile_blocks(x, profiler)
        total_blocks += num_blocks
        convert_cost += convert_time
        profile_cost += profile_time
        
    logger.info(f"Convert cost for {total_blocks} blocks: {convert_cost} s")
    logger.info(f"Profile cost for {total_blocks} blocks: {profile_cost} s")

else:
    if args.net == "ofa_mbv3_d234_e346_k357_w1.0":
        ofa_network.count_all_subnets()