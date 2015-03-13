#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_with_exemplars_part_2/test_mednet_relu_1_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./.build_debug/tools/caffe.bin train \
  --solver=projects/med_embedding_with_exemplars_part_2/mednet_relu_1_solver_on_test_data.prototxt \
  --gpu=-1 \
  --weights=projects/med_embedding_with_exemplars_full/mednet_relu_6_exemplar_full_4096_iter_12000.caffemodel
#  --weights=projects/med_embedding_with_exemplars_full/mednet_relu_9_exemplar_full_4096_ret_iter_14000.caffemodel
#  --weights=projects/med_embedding_with_exemplars_full/mednet_relu_9_exemplar_full_4096_ret_iter_14000.caffemodel
#  --weights=projects/med_embedding_with_exemplars_full/mednet_relu_7_exemplar_full_4096_ret_iter_16000.caffemodel
#  --weights=projects/med_embedding_with_exemplars_full/mednet_relu_9_exemplar_full_4096_ret_iter_16000.caffemodel
#  --weights=projects/med_embedding_with_exemplars_full/mednet_relu_7_exemplar_full_4096_ret_iter_16000.caffemodel
#  --weights=projects/med_embedding_with_exemplars_part_2/mednet_relu_2_exemplar_p2_4096_iter_20000.caffemodel
#  --weights=projects/med_embedding_with_exemplars_part_2/mednet_relu_1_exemplar_p2_4096_iter_30000.caffemodel
#  --snapshot=projects/med_embedding_with_exemplars_part_2/mednet_relu_1_exemplar_p2_4096_iter_12000.solverstate
