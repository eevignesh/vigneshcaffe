#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_max_margin/test_mednet_para_train_256_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./build/tools/caffe train \
  --solver=projects/med_embedding_max_margin/test_mednet_para_solver.prototxt \
  --snapshot=projects/med_embedding_max_margin/mednet_para_256_w0_iter_100000.solverstate \
  --gpu=1
