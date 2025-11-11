#include "algo/matting/inference_mat.h"

namespace matting
{
    Matter::Matter(const std::string &engine_or_onnx, std::shared_ptr<spdlog::logger> &logger, int device_id)
        : logger_(logger), device_id_(device_id)
    {
        if (!file_exist(engine_or_onnx))
            throw std::runtime_error("Error: The engine or onnx file does not exist: " + engine_or_onnx);

        if (ends_with(engine_or_onnx, ".onnx"))
            init_onnx(engine_or_onnx);
        else
            throw std::runtime_error("Error: only support onnx file: " + engine_or_onnx);

        // cv::Mat dummy_frame = cv::Mat(512, 512, CV_8UC3, cv::Scalar(0, 0, 0));
        // logger_->info("[Matter] warmup...");
        // for (int i = 0; i < 2; i++)
        //     run(dummy_frame);
        logger_->info("[Matter] init matting success...");
        std::cout << "[Matter] init matting success..." << std::endl;
    }

    void Matter::init_onnx(std::string onnx_path)
    {
        logger_->info("[Matter] init_onnx with onnx file: {}", onnx_path);
        onnx_engine_ = std::make_unique<ONNXEngine>(onnx_path, device_id_);

        input_h_ = onnx_engine_->input_height_;
        input_w_ = onnx_engine_->input_width_;
        logger_->info("[Matter] input_h: {} input_w: {}", input_h_, input_w_);

        const auto &output_dims = onnx_engine_->output_node_dims_;
        int i = 0;
        for (const auto &dim : output_dims)
        {
            logger_->info("[Matter] output{}: {}", i, print_1d_vector(dim));
            i++;
        }
        std::cout << "[Matter] init_onnx with onnx file: " << onnx_path << std::endl;
    }

    cv::Mat Matter::resize_and_pad(const cv::Mat &input, size_t height, size_t width, const cv::Scalar &bgcolor)
    {
        float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
        int unpad_w = r * input.cols;
        int unpad_h = r * input.rows;
        cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        cv::resize(input, re, re.size());
        cv::Mat out(height, width, CV_8UC3, bgcolor);
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        return out;
    }

    std::vector<std::vector<cv::Mat>> Matter::pre_process(const cv::Mat &input_rgb)
    {
        cv::Mat rgb;
        if (input_rgb.channels() == 4)
            cv::cvtColor(input_rgb, rgb, cv::COLOR_BGRA2RGB);
        else
            rgb = input_rgb;

        cv::Mat resized;
        if (input_rgb.cols == input_w_ && input_rgb.rows == input_h_)
            resized = std::move(rgb);
        else
        {
            cv::resize(rgb, resized, cv::Size(input_w_, input_h_));
            // resized = resize_and_pad(bgr, input_h_, input_w_);
        }

        // cv::imwrite("debug_resized.jpg", resized);
        std::vector<cv::Mat> input{std::move(resized)};
        std::vector<std::vector<cv::Mat>> inputs{std::move(input)}; // [input][batch][image]

        return inputs;
    }

    ObjectMatting Matter::run(const cv::Mat &input_rgb)
    {
        auto t1 = time_now();
        ori_img_h_ = input_rgb.rows;
        ori_img_w_ = input_rgb.cols;
        std::vector<std::vector<cv::Mat>> input_resize = pre_process(input_rgb);
        auto t2 = time_now();

        std::vector<std::vector<std::vector<float>>> batch_feature_map; // [batch_index][output_index][feature]
        bool succ = false;
        if (onnx_engine_)
        {
            batch_feature_map.resize(1);
            succ = onnx_engine_->run(input_resize[0], batch_feature_map[0]);
        }
        else
            std::runtime_error("Error: No valid engine found.");

        if (!succ)
            throw std::runtime_error("Error: Unable to run.");

        auto t3 = time_now();
        if (batch_feature_map.size() != 1 || batch_feature_map[0].size() != 1)
            std::cout << "The feature vector has incorrect dimensions!" << std::endl;

        ObjectMatting res;
        process_alpha(res.alpha, batch_feature_map[0][0]);

        auto t4 = time_now();
        // std::cout << "[Matter] frame " << input_rgb.rows << "x" << input_rgb.cols << "x" << input_rgb.channels() << " -> " << input_resize[0][0].rows << "x" << input_resize[0][0].cols
        //           << "x" << input_resize[0][0].channels() << " time pre_process: " << time_cost(t2, t1) << " inference: " << time_cost(t3, t2) << " post_process: " << time_cost(t4, t3)
        //           << " matting_total: " << time_cost(t4, t1) << std::endl;
        logger_->info("[Matter] frame {}x{}x{} -> {}x{}x{} time pre_process: {} inference: {} post_process: {}, matting_total: {}", input_rgb.rows, input_rgb.cols, input_rgb.channels(),
                      input_resize[0][0].rows, input_resize[0][0].cols, input_resize[0][0].channels(), time_cost(t2, t1), time_cost(t3, t2), time_cost(t4, t3), time_cost(t4, t1));
        batch_feature_map.clear();
        return res;
    }

    void Matter::process_alpha(cv::Mat &alpha, std::vector<float> &feature_map)
    {
        cv::Mat alpha_float(input_h_, input_w_, CV_32FC1, feature_map.data());

        // double min_value, max_value;
        // cv::Point min_location, max_location;
        // cv::minMaxLoc(alpha_float, &min_value, &max_value, &min_location, &max_location);
        // std::cout << "feature_map size: " << feature_map.size() << std::endl;
        // std::cout << "min value: " << min_value << std::endl;
        // std::cout << "max value: " << max_value << std::endl;
        // alpha_float.setTo(1, alpha_float > 1);
        // alpha_float.setTo(0, alpha_float < 0);

        alpha_float *= 255.f;
        alpha_float.convertTo(alpha, CV_8UC1);
        // cv::imwrite("debug_alpha_resized.png", alpha);
        cv::resize(alpha, alpha, cv::Size(ori_img_w_, ori_img_h_));
    }
};
