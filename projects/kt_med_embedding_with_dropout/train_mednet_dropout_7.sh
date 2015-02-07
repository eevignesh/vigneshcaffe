#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_with_dropout/train_mednet_dropout_7_512_relu_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./build/tools/caffe train --solver=projects/med_embedding_with_dropout/mednet_dropout_7_solver.prototxt --gpu=2
