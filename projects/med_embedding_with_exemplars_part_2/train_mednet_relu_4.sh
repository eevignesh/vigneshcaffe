#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_with_exemplars_part_2/train_mednet_relu_4_4096_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./.build_debug/tools/caffe.bin train \
  --solver=projects/med_embedding_with_exemplars_part_2/mednet_relu_4_solver.prototxt --gpu=1 \
  --weights=projects/med_embedding_with_relu/mednet_pair_relu_1_4096_iter_12000.caffemodel
