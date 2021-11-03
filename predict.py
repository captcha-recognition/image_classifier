from os import confstr
from re import A
import re
from tqdm.cli import main
from tqdm.std import TRLock
from models.classifier import Classifier
import torch.nn as nn
import torch.nn.functional as F
import torch
from dataset import test_loader, train_loader
from tqdm import tqdm
from argparse import ArgumentParser

def main(args):
    net = Classifier(args.input_shape, args.out_channel, None,train = False)
    test_it = test_loader(args.test_path,multi = args.multi,keep_ratio=False)
    acc = 0
    for imgs, labels in tqdm(test_it):
        preds = net.predict(imgs)
        labels= labels.reshape((-1))
        c = (preds == labels)
        acc += torch.sum(c).item()
    print(f'{acc}/{len(test_it.dataset)}, {acc/len(test_it.dataset)}')
  


if __name__ == '__main__':
    args = ArgumentParser(description= 'Images Classifier')
    args.add_argument('--input_shape', '--i', type=tuple,default=(3,32,100),help='The shape of input image')
    args.add_argument('--out_channel','--o', type=int,default=2,help='the channel of output')
    args.add_argument('--test_path','--t',type=str, required=True ,help='train_path')
    args.add_argument('--multi','--mu',type=bool,default=True,help='multi or not')
    args.add_argument('--checkpoints','--c',type=str, required=True ,help='checkpoints')
    main(args.parse_args())