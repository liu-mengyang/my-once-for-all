import argparse
import os

import horovod.torch as hvd
import torch

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager.run_config import ImagenetRunConfig
from ofa.imagenet_classification.run_manager.run_manager import RunManager
from ofa.model_zoo import ofa_net
from ofa.nas.accuracy_predictor.acc_dataset import AccuracyDataset


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-p", "--path", help="The path of imagenet", type=str, default="/dataset/imagenet"
    )
    arg_parser.add_argument(
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
    arg_parser.add_argument(
        "-b",
        "--batch-size",
        help="The batch on every device for validation",
        type=int,
        default=128,
    )
    arg_parser.add_argument("-g", "--gpu", help="The gpu(s) to use", type=str, default="all")
    arg_parser.add_argument("-ns", "--num_samples", type=int, default=1000)
    arg_parser.add_argument("--image_size_list", type=str, default=None)
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    ImagenetDataProvider.DEFAULT_PATH = args.path
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    
    ofa_network = ofa_net(args.net, pretrained=True)
    run_config = ImagenetRunConfig(test_batch_size=args.batch_size,n_worker=16)
    
    run_manager = RunManager(
        ".tmp/eval_subnet",
        ofa_network,
        run_config,
        init=False
    )
    run_manager.save_config()
    arch = args.net
    image_size_list = args.image_size_list
    if arch == "ofa_mbv3_d234_e346_k357_w1.0":
        path = "mbv3w10"
        if image_size_list is None:
            image_size_list = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                            168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
                            208, 212, 216, 220, 224]
    elif arch == "ofa_mbv3_d234_e346_k357_w1.2":
        path = "mbv3w12"
        if image_size_list is None:
            image_size_list = [160, 164, 168, 172, 176, 180, 184, 188, 192, 196,
                            200, 204, 208, 212, 216, 220, 224]
    elif arch == "ofa_proxyless_d234_e346_k357_w1.3":
        path = "proxylessnet"
        if image_size_list is None:
            image_size_list = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                            168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
                            208, 212, 216, 220, 224]
    elif arch == "ofa_resnet50":
        path = "resnet50"
        if image_size_list is None:
            image_size_list = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                            168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
                            208, 212, 216, 220, 224]
    else:
        raise NotImplementedError
    
    acc_dataset = AccuracyDataset(path, args.path)
    acc_dataset.build_acc_dataset(run_manager,
                                  ofa_network,
                                  n_arch=args.num_samples,
                                  image_size_list=image_size_list)
    
if __name__ == "__main__":
    main()