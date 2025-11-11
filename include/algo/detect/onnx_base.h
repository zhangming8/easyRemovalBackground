#pragma once
#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

#include "utils/util.h"

class ONNXEngine
{
public:
    ONNXEngine(const std::string onnx_path, int device_id = 0);
    ~ONNXEngine();

    std::vector<std::vector<long long int>> input_node_dims_, output_node_dims_;
    int input_height_, input_width_, input_channel_, batch_size_;
    std::vector<std::string> input_names_, output_names_;
    bool run(const std::vector<cv::Mat> &inputs, std::vector<std::vector<float>> &batch_feature_map);

private:
    cv::Mat blob_;
    Ort::Env env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Base ONNX Engine");
    Ort::Session *ort_session_ = nullptr;
    Ort::SessionOptions session_options_ = Ort::SessionOptions();
    std::vector<long long int> input_img_shape_;
    Ort::MemoryInfo memory_info_handler_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};
