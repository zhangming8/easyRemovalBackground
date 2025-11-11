#pragma once

#include "algo/detect/onnx_base.h"

namespace matting
{
    class Matter
    {
    public:
        Matter(const std::string &engine_or_onnx, std::shared_ptr<spdlog::logger> &logger, int device_id = 0);
        ObjectMatting run(const cv::Mat &input_bgr);
        int input_h_;
        int input_w_;

    private:
        bool normalize_ = false;
        float ori_img_w_, ori_img_h_;
        float img_ratio_;
        int alpha_h_, alpha_w_;
        int pose_feat_h_, pose_feat_w_, pose_channel_, pose_stride_, device_id_;
        std::shared_ptr<spdlog::logger> logger_;
        std::unique_ptr<ONNXEngine> onnx_engine_ = nullptr;

        std::vector<std::vector<cv::Mat>> pre_process(const cv::Mat &input_bgr);
        void process_alpha(cv::Mat &alpha, std::vector<float> &feature_map);
        cv::Mat resize_and_pad(const cv::Mat &input, size_t height, size_t width, const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));
        void init_trt(std::string engine_path);
        void init_onnx(std::string onnx_path);
    };

} // namespace matting
