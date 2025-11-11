
#include <iostream>
#include <fstream>
#include <cstdlib>

#include "utils/util.h"
#include "algo/detect/onnx_base.h"

ONNXEngine::ONNXEngine(const std::string onnx_path, int device_id)
{
    if (device_id >= 0)
    {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = device_id;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "Using CUDA Execution Provider with device ID: " << device_id << std::endl;
    }
    else
    {
        std::cout << "Using CPU Execution Provider." << std::endl;
    }
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    std::cout << "Loading ONNX model from path: " << onnx_path << std::endl;
    std::wstring model_path = std::wstring(onnx_path.begin(), onnx_path.end()); // Convert to wide string when using Windows
    ort_session_ = new Ort::Session(env_, model_path.c_str(), session_options_);

    size_t num_input_nodes = ort_session_->GetInputCount();
    size_t num_output_nodes = ort_session_->GetOutputCount();
    if (num_input_nodes == 0 || num_output_nodes == 0)
    {
        std::cerr << "Error: No input or output nodes found in the ONNX model." << std::endl;
        exit(-1);
    }

    Ort::AllocatorWithDefaultOptions allocator;

    for (int i = 0; i < num_input_nodes; i++)
    {
        Ort::AllocatedStringPtr input_name_ptr = ort_session_->GetInputNameAllocated(i, allocator);

        auto name = std::string(input_name_ptr.get());
        input_names_.push_back(name);

        Ort::TypeInfo input_type_info = ort_session_->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<long long int> input_dims = input_tensor_info.GetShape();
        input_node_dims_.push_back(input_dims);
        std::cout << "input" << i << " name: '" << name << "' shape: " << print_1d_vector(input_dims) << std::endl;
    }

    batch_size_ = input_node_dims_[0][0];
    input_channel_ = input_node_dims_[0][1];
    input_height_ = input_node_dims_[0][2];
    input_width_ = input_node_dims_[0][3];
    std::cout << "input shape: " << batch_size_ << "x" << input_channel_ << "x" << input_height_ << "x" << input_width_ << std::endl;
    input_img_shape_ = {batch_size_, input_channel_, input_height_, input_width_};

    for (int i = 0; i < num_output_nodes; i++)
    {
        Ort::AllocatedStringPtr output_name_ptr = ort_session_->GetOutputNameAllocated(i, allocator);
        auto name = std::string(output_name_ptr.get());
        output_names_.push_back(name);

        Ort::TypeInfo output_type_info = ort_session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<long long int> output_dims = output_tensor_info.GetShape();
        output_node_dims_.push_back(output_dims);
        std::cout << "output" << i << " name: '" << name << "' shape: " << print_1d_vector(output_dims) << std::endl;
    }
    std::cout << "ONNX model loaded successfully." << std::endl;
}

ONNXEngine::~ONNXEngine()
{
    if (ort_session_)
    {
        delete ort_session_;
        ort_session_ = nullptr;
    }
}

bool ONNXEngine::run(const std::vector<cv::Mat> &inputs, std::vector<std::vector<float>> &batch_feature_map)
{
    // input format [batch][input][image]
    // batch_feature_map format [batch][output][feature_vector]
    if (inputs.empty() || inputs[0].empty())
    {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }
    if (inputs[0].rows != input_height_ || inputs[0].cols != input_width_)
    {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Input image size does not match the model's expected input size!" << std::endl;
        std::cout << "Expected size: " << input_width_ << "x" << input_height_ << ", Provided size: " << inputs[0].cols << "x" << inputs[0].rows << std::endl;
        return false;
    }

    // cv::Mat blob = cv::dnn::blobFromImages(inputs, 1.0, cv::Size(input_width_, input_height_), cv::Scalar(0, 0, 0), false, false); // b,3,h,w: has normalized in network, so no need to normalize here
    cv::Mat blob = cv::dnn::blobFromImages(inputs);
    // std::cout << "Blob shape: " << blob.size[0] << " x " << blob.size[1] << " x " << blob.size[2] << " x " << blob.size[3] << " type " << blob.type() << std::endl;

    std::vector<Ort::Value> input_tensor = {};
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info_handler_, blob.ptr<float>(), blob.total(), input_img_shape_.data(), input_img_shape_.size()));
    // std::cout << "Input tensor created successfully." << print_1d_vector(input_tensor[0].GetTensorTypeAndShapeInfo().GetShape()) << " type: " << input_tensor[0].GetTensorTypeAndShapeInfo().GetElementType() << std::endl;

    std::vector<const char *> input_names = {}, output_names = {};
    for (auto &name : input_names_)
        input_names.push_back(name.c_str());
    for (auto &name : output_names_)
        output_names.push_back(name.c_str());

    Ort::RunOptions run_options;
    auto ort_outputs = ort_session_->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensor.data(), 1, output_names.data(), output_names.size());

    for (auto &output : ort_outputs)
    {
        float *pdata = output.GetTensorMutableData<float>();
        size_t total_len = output.GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> feature_map(pdata, pdata + total_len);
        batch_feature_map.push_back(feature_map);

        // std::cout << "Output tensor shape: " << print_1d_vector(output.GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
        // std::cout << "Output feature map size: " << feature_map.size() << std::endl;
    }
    return true;
}
