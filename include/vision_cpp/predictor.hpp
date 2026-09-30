#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace vision_cpp
{

struct Detection
{
    int class_id = -1;

    std::string class_name;

    float confidence = 0.0f;

    cv::Rect box;

    cv::Mat mask;

    float distance = -1.0f;

    cv::Point3f point_3d;

    bool has_depth = false;
};

class Predictor
{
public:

    Predictor(
        const std::string &model_path,
        const std::string &device = "CPU",
        float confidence = 0.7f,
        int input_size = 640
    );

    std::vector<Detection> predict(
        const cv::Mat &frame
    );

    const std::vector<std::string> &get_class_names() const;

private:

    ov::Core core_;

    ov::CompiledModel compiled_model_;

    ov::InferRequest infer_request_;

    std::vector<std::string> class_names_;

    float confidence_;

    int input_size_;

    cv::Mat preprocess(
        const cv::Mat &frame
    );

    std::vector<Detection> postprocess(
        const cv::Mat &frame,
        const ov::Tensor &detection_output,
        const ov::Tensor &proto_output
    );

    cv::Mat process_mask(
        const std::vector<float> &mask_coefficients,
        const ov::Tensor &proto_output,
        const cv::Rect &box,
        const cv::Size &image_size
    );

    void initialize_classes();
};

}