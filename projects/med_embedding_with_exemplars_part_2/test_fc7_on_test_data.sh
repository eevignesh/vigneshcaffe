#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_with_exemplars_part_2/mednet_fc7_test_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR .build_debug/tools/caffe.bin train \
  --solver=projects/med_embedding_with_exemplars_part_2/mednet_fc7_solver_on_test_data.prototxt \
  --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  --gpu=-1
