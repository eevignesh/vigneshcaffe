#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_normalized_euclid/train_mednet_64_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./build/tools/caffe train --solver=projects/med_embedding_normalized_euclid/mednet_solver.prototxt
