#!/usr/bin/bash
online=$1
checkpoints='checkpoints/20211103_12026_model.pt'
if [ ${online} == "linux" ];then
   python predict.py --t '/Disk/hsj/dataset/test' --c ${checkpoints}
else
   python predict.py --t '/Users/sjhuang/Documents/docs/dataset/test' --c ${checkpoints} --mu True
fi
