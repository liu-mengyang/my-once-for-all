# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import random
import argparse

import torch
import torch.nn as nn

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.model_zoo import ofa_net
from ofa.utils import MyModule
from ofa.logger import logger


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", help="The path of imagenet", type=str, default="/dataset/imagenet"
)
parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
parser.add_argument(
    "-b",
    "--batch-size",
    help="The batch on every device for validation",
    type=int,
    default=100,
)
parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=20)
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
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path

ofa_network = ofa_net(args.net, pretrained=True)
run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

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

for i in range(int(args.num_samples)):
    """ Randomly sample a sub-network, 
        you can also manually set the sub-network using: 
            ofa_network.set_active_subnet(ks=7, e=6, d=4) 
    """
    logger.info("Sampling ...")
    ofa_network.sample_active_subnet()
    logger.info("Sampled")
    logger.info("Building ...")
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    logger.info("Built")
    
    """ Test sampled subnet 
    """
    logger.info("Preparing ...")
    run_manager = RunManager(".tmp/eval_subnet", subnet, run_config, init=False)
    # assign image size: 128, 132, ..., 224
    img_size = random.choice([128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224])
    run_config.data_provider.assign_active_img_size(img_size)
    run_manager.reset_running_statistics(net=subnet)
    logger.info("Prepared")

    print("Test random subnet:")
    print(subnet.module_str)

    logger.info("Validating ...")
    loss, (top1, top5) = run_manager.validate(net=subnet)
    logger.info("Vlidated")
    logger.info("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (loss, top1, top5))
    
    # Output information
    register_info_fw_hook(subnet)
    x = torch.randn(1, 3, img_size, img_size).cuda()
    y = subnet(x)
    print(y.shape)