#!/usr/bin/env sh

LOG_DIR="/data2/vigneshr/vigneshcaffe/projects/med_embedding_discrete_words/train_mednet_para_256_hard_negs_reweight_log_dir/"
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./build/tools/caffe train --solver=projects/med_embedding_discrete_words/mednet_para_hard_negs_reweight_solver.prototxt --gpu=2
