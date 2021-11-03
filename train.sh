#!/usr/bin/bash

save_path='checkpoints'
if [ ! -d ${save_path} ];then
   mkdir ${save_path}
   echo '${save_path} create success!'
else
   echo '${save_path} exits'
fi

python train.py --t '/Users/sjhuang/Documents/docs/dataset/train' --s ${save_path}
