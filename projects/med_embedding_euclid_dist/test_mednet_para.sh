#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_euclid_dist/test_mednet_para_64_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./build/tools/caffe train \
  --solver=projects/med_embedding_euclid_dist/test_mednet_para_solver.prototxt \
  --snapshot=projects/med_embedding_euclid_dist/mednet_para_64_iter_10000.solverstate
