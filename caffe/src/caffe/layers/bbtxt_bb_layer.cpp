#include <algorithm>
#include <vector>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/bbtxt_bb_layer.hpp"

namespace caffe {


template <typename Dtype>
BBTXTBBLayer<Dtype>::BBTXTBBLayer (const LayerParameter &param)
    : NeuronLayer<Dtype>(param)
{
    CHECK(param.has_bbtxt_bb_param()) << "BBTXTScalingParameter is mandatory!";
    CHECK(param.bbtxt_bb_param().has_ideal_size()) << "Ideal size is mandatory!";
    CHECK(param.bbtxt_bb_param().has_downsampling()) << "Downsampling is mandatory!";
    CHECK(param.bbtxt_bb_param().has_path_obj_g_bb_xml()) << "P(OBJ_GT|BB) is mandatory!";

    // Load P(CAR_GT|BB) WxH
    cv::FileStorage fs(this->layer_param_.bbtxt_bb_param().path_obj_g_bb_xml(), cv::FileStorage::READ);
    fs["acc_x" + std::to_string(this->layer_param_.bbtxt_bb_param().downsampling())] >> this->_obj_g_bb;
    fs.release();
}


template <typename Dtype>
void BBTXTBBLayer<Dtype>::Forward_cpu (const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
{
    const Dtype ideal_size   = this->layer_param_.bbtxt_bb_param().ideal_size();
    const Dtype downsampling = this->layer_param_.bbtxt_bb_param().downsampling();

    // For each image in the batch
    for (int b = 0; b < bottom[0]->shape(0); ++b)
    {
        cv::Mat acc_prob(bottom[0]->shape(2), bottom[0]->shape(3), CV_32FC1, bottom[0]->mutable_cpu_data() + bottom[0]->offset(b, 0));
        cv::Mat acc_xmin(bottom[0]->shape(2), bottom[0]->shape(3), CV_32FC1, bottom[0]->mutable_cpu_data() + bottom[0]->offset(b, 1));
        cv::Mat acc_ymin(bottom[0]->shape(2), bottom[0]->shape(3), CV_32FC1, bottom[0]->mutable_cpu_data() + bottom[0]->offset(b, 2));
        cv::Mat acc_xmax(bottom[0]->shape(2), bottom[0]->shape(3), CV_32FC1, bottom[0]->mutable_cpu_data() + bottom[0]->offset(b, 3));
        cv::Mat acc_ymax(bottom[0]->shape(2), bottom[0]->shape(3), CV_32FC1, bottom[0]->mutable_cpu_data() + bottom[0]->offset(b, 4));

        cv::Mat acc_prob_out(top[0]->shape(2), top[0]->shape(3), CV_32FC1, top[0]->mutable_cpu_data() + top[0]->offset(b, 0));

        // Extended confidence
        for (int i = 0; i < bottom[0]->shape(2); ++i)
        {
            for (int j = 0; j < bottom[0]->shape(3); ++j)
            {
                double w = acc_xmax.at<float>(i, j) - acc_xmin.at<float>(i, j);
                double h = acc_ymax.at<float>(i, j) - acc_ymin.at<float>(i, j);

                int col = std::round((w - 0) / 2 * this->_obj_g_bb.cols);
                int row = std::round((h - 0) / 2 * this->_obj_g_bb.rows);

                if (col >= 0 && row >= 0 && col < this->_obj_g_bb.cols && row < this->_obj_g_bb.rows
                        && this->_obj_g_bb.template at<double>(row, col) == this->_obj_g_bb.template at<double>(row, col)) // Test NaN
                {
                    // Extend the confidence with P(OBJ_GT|BB)
                    acc_prob_out.at<float>(i, j) = acc_prob.at<float>(i, j) + this->_obj_g_bb.template at<double>(row, col);
                }
                else
                {
                    acc_prob_out.at<float>(i, j) = acc_prob.at<float>(i, j);
                }
            }
        }

        // For each channel
        for (int c = 1; c < 5; ++c)
        {
            const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(b, c);
            Dtype* top_data          = top[0]->mutable_cpu_data() + top[0]->offset(b, c);

            for (int i = 0; i < bottom[0]->shape(2); ++i)
            {
                for (int j = 0; j < bottom[0]->shape(3); ++j)
                {
                    // Convert the local normalized coordinate into a global unnormalized value in pixels
                    if (c == 1 || c == 3)
                    {
                        // xmin, xmax
                        *top_data = downsampling*j + ideal_size * (*bottom_data - Dtype(0.5f));
                    }
                    else // (c == 2 || c == 4)
                    {
                        // ymin, ymax
                        *top_data = downsampling*i + ideal_size * (*bottom_data - Dtype(0.5f));
                    }

                    bottom_data++;
                    top_data++;
                }
            }
        }
    }
}


template <typename Dtype>
void BBTXTBBLayer<Dtype>::Backward_cpu (const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype>*> &bottom)
{
    CHECK(false) << "BBTXTBBLayer implements only the forward pass!";
}


// ----------------------------------------  LAYER INSTANTIATION  ---------------------------------------- //

#ifdef CPU_ONLY
//STUB_GPU(BBTXTBBLayer);
#endif

INSTANTIATE_CLASS(BBTXTBBLayer);
REGISTER_LAYER_CLASS(BBTXTBB);


}  // namespace caffe
