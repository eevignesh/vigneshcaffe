#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_with_exemplars_part_2/test_mednet_relu_2_train_dev_4096_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./build/tools/caffe train \
  --solver=projects/med_embedding_with_exemplars_part_2/mednet_relu_2_test_train_dev_solver.prototxt --gpu=2 \
  --weights=projects/med_embedding_with_exemplars_part_2/mednet_relu_2_exemplar_4096_iter_22000.caffemodel
