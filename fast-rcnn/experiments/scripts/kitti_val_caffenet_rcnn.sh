#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/kitti_val_caffenet_rcnn.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --network caffenet \
  --weights data/imagenet_models/caffenet.npy \
  --imdb kitti_train \
  --cfg experiments/cfgs/kitti_rcnn.yml

time ./tools/test_net.py --gpu $1 \
  --network caffenet \
  --weights output/kitti/kitti_train/caffenet_fast_rcnn_kitti_iter_40000.ckpt \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rcnn.yml
