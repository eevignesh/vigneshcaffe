#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_euclid_dist/train_mednet_64_full_vec_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./build/tools/caffe train --solver=projects/med_embedding_euclid_dist/mednet_full_vec_solver.prototxt
