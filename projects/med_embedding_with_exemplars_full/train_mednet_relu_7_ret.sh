#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_with_exemplars_full/train_mednet_relu_7_4096_ret_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./.build_debug/tools/caffe.bin train \
  --solver=projects/med_embedding_with_exemplars_full/mednet_relu_7_solver_ret.prototxt --gpu=3 \
  --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
#  --weights=projects/med_embedding_with_relu/mednet_pair_relu_1_4096_iter_12000.caffemodel
#  --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
#  --weights=projects/med_embedding_with_relu/mednet_pair_relu_1_4096_iter_12000.caffemodel
