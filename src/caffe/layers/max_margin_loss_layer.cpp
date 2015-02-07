#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MaxMarginLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->Reshape(1,1,1,1);
  }
}

template <typename Dtype>
void MaxMarginLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_true_data = bottom[0]->cpu_data();
  const Dtype* bottom_bogus_data = bottom[1]->cpu_data();

  // For temporary storage
  Dtype* bottom_bogus_diff = bottom[0]->mutable_cpu_diff();

  int count = bottom[0]->count();

  caffe_sub(count, bottom_true_data, bottom_bogus_data, bottom_bogus_diff);

  //float maxval = -1;
  float num_violations = 0;
  for (int i = 0; i < count; ++i) {
    /*if (bottom_bogus_diff[i] > 1) {
      LOG(INFO) << bottom_true_data[i] << ":" << bottom_bogus_data[i];
    }*/

    if (bottom_bogus_diff[i] < 0) {
      num_violations++;
    }

    bottom_bogus_diff[i] = std::max(Dtype(0), 1 - bottom_bogus_diff[i]);
    
    
    /*if (bottom_bogus_diff[i] > maxval) {
      maxval = bottom_bogus_diff[i];
    }*/
    //LOG(INFO) << bottom_bogus_diff[i];
  }

  //LOG(INFO) << "Max-val: " << maxval;

  Dtype* loss = (*top)[0]->mutable_cpu_data();
  switch (this->layer_param_.max_margin_loss_param().norm()) {
  case MaxMarginLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, bottom_bogus_diff) / count;
    break;
  case MaxMarginLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_bogus_diff, bottom_bogus_diff) / count;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }

  if (top->size() > 1) {
    Dtype* nv = (*top)[1]->mutable_cpu_data();
    nv[0] = Dtype(num_violations);
  }
}

template <typename Dtype>
void MaxMarginLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  Dtype* bottom_bogus_diff = (*bottom)[1]->mutable_cpu_diff();
  Dtype* bottom_true_diff = (*bottom)[0]->mutable_cpu_diff();

  int count = (*bottom)[0]->count();

  // Bogus diff already has the difference

  if (propagate_down[1] || propagate_down[0]) {

    caffe_sub(count, (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(), bottom_bogus_diff);

    for (int i = 0; i < count; ++i) {
      bottom_bogus_diff[i] = std::max(Dtype(0), 1 - bottom_bogus_diff[i]);
    }


    const Dtype loss_weight = top[0]->cpu_diff()[0];
    //LOG(INFO) << "loss weight: " << loss_weight;
    //loss_weight = Dtype(1.0);
    float grad_norm = 0.0;
    switch (this->layer_param_.max_margin_loss_param().norm()) {
    case MaxMarginLossParameter_Norm_L1:
      //caffe_cpu_sign(count, bottom_bogus_diff, bottom_bogus_diff);
      for (int i = 0; i < count; ++i) {
        if (bottom_bogus_diff[i] > Dtype(0)) {
          bottom_bogus_diff[i] = Dtype(1);
        }
      }
      caffe_scal(count, loss_weight / count, bottom_bogus_diff);
      break;
    case MaxMarginLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / count, bottom_bogus_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
    
    /*float maxval = -1;
    for (int i = 0; i < count; ++i) {
      float alpha = static_cast<float>(bottom_bogus_diff[i]);
      //LOG(INFO) << alpha << ":" << loss_weight << ":" << (*bottom)[1]->cpu_data()[i] << ":" << (*bottom)[0]->cpu_data()[i];
      if (alpha > maxval) {
        maxval = alpha;
      }
      grad_norm += alpha*alpha;
    }
    LOG(INFO) << "Grad-norm: " << grad_norm/count << ", max-grad: " << maxval;
    */
  }

  if (propagate_down[0]) {
    caffe_cpu_axpby(count, Dtype(-1), bottom_bogus_diff, Dtype(0), bottom_true_diff);
  }

}

INSTANTIATE_CLASS(MaxMarginLossLayer);

}  // namespace caffe
