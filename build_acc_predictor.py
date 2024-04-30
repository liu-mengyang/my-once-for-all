import argparse

from ofa.nas.accuracy_predictor.acc_predictor import AccuracyPredictor
from ofa.nas.accuracy_predictor.arch_encoder import MobileNetArchEncoder, ResNetArchEncoder
from ofa.nas.accuracy_predictor.acc_dataset import AccuracyDataset
from ofa.nas.accuracy_predictor.acc_trainer import AccPredictorTrainer


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", type=str)
    arg_parser.add_argument("--save_name", type=str, default="test")
    arg_parser.add_argument("--num_epochs", type=int, default=200)
    arg_parser.add_argument('--pretrained_model_path', type=str, default=None)
    arg_parser.add_argument('--eval', action='store_true', default=False)
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    
    dataset = AccuracyDataset(args.path)
    image_size_list = [128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                           168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
                           208, 212, 216, 220, 224]
    if path == "resnet50":
        arch_encoder = ResNetArchEncoder(image_size_list)
    elif path == "mbv3w12":
        image_size_list = [160, 164, 168, 172, 176, 180, 184, 188, 192, 196,
                           200, 204, 208, 212, 216, 220, 224]
        arch_encoder = MobileNetArchEncoder(image_size_list)
    else:
        arch_encoder = MobileNetArchEncoder(image_size_list)
    train_dataloader, eval_dataloader, base_acc = dataset.build_acc_data_loader()
    print(base_acc)
    predictor = AccuracyPredictor(arch_encoder)
    
    trainer = AccPredictorTrainer(predictor,
                                  train_dataloader,
                                  eval_dataloader,
                                  args.save_name,
                                  args.num_epochs)
    
    if args.pretrained_model_path:
        trainer.load(args.pretrained_model_path)
    
    if args.eval:
        trainer.evaluate()
        return
    
    trainer.train(base_acc)
    trainer.save()
    
    
if __name__ == "__main__":
    main()