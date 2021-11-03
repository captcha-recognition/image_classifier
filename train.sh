#!/usr/bin/bash
online=$1
save_path='checkpoints'
if [ ! -d ${save_path} ];then
   mkdir ${save_path}
   echo '${save_path} create success!'
else
   echo '${save_path} exits'
fi

if [ ${online} == "linux" ];then
   python train.py --t '/Disk/hsj/dataset/train' --s ${save_path}
else
   python train.py --t '/Users/sjhuang/Documents/docs/dataset/train' --s ${save_path}
fi
