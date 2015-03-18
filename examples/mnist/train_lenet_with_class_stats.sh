#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/examples/mnist/train_lenet_with_class_stats_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./build/tools/caffe train --solver=examples/mnist/lenet_solver_with_class_stats.prototxt
