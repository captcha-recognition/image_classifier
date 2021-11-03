from os import confstr
from re import A
import re
from tqdm.cli import main
from models.classifier import Classifier
import torch.nn as nn
import torch.nn.functional as F
import torch
from dataset import train_loader
from tqdm import tqdm
from argparse import ArgumentParser

def main(args):
    net = Classifier(args.input_shape, args.out_channel, args.save_path)
    Optim = torch.optim.SGD(net.parameters(),lr = args.lr,momentum= args.momentum)
    Loss = nn.CrossEntropyLoss()
    train_it, val_it = train_loader(args.train_path,args.multi,keep_ratio=False)
    epochs = args.epochs
    for epoch in tqdm(range(epochs),desc = 'Training', mininterval = 3):
        if net.runing():
            net.train(Optim,Loss,epoch,train_it)
            net.eval(Loss,epoch,val_it)
            net.desc()
  


if __name__ == '__main__':
    args = ArgumentParser(description= 'Images Classifier')
    args.add_argument('--input_shape', '--i', type=tuple,default=(3,32,100),help='The shape of input image')
    args.add_argument('--out_channel','--o', type=int,default=2,help='the channel of output')
    args.add_argument('--lr','--lr', type=float,default=1e-3,help='learning rate')
    args.add_argument('--momentum','--m',type=float,default=0.9,help='momentum')
    args.add_argument('--epochs','--e',type=int,default=100,help='epochs')
    args.add_argument('--train_path','--t',type=str, required=True ,help='train_path')
    args.add_argument('--multi','--mu',type=bool,default=True,help='multi or not')
    args.add_argument('--save_path','--s',type=str, required=True ,help='save_path')
    main(args.parse_args())