#include "vision_cpp/predictor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace vision_cpp
{

Predictor::Predictor(
    const std::string &model_path,
    const std::string &device,
    float confidence,
    int input_size)
    : confidence_(confidence),
      input_size_(input_size)
{
    std::cout
        << "[INFO] Loading OpenVINO model: "
        << model_path
        << std::endl;

    core_ = ov::Core();

    auto model =
        core_.read_model(model_path);

    compiled_model_ =
        core_.compile_model(
            model,
            device,
            ov::hint::performance_mode(
                ov::hint::PerformanceMode::LATENCY
            )
        );

    infer_request_ =
        compiled_model_.create_infer_request();

    initialize_classes();

    std::cout
        << "[INFO] OpenVINO device: "
        << device
        << std::endl;

    std::cout
        << "[INFO] Input size: "
        << input_size_
        << std::endl;
}


void Predictor::initialize_classes()
{
    class_names_ = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush"
    };
}


cv::Mat Predictor::preprocess(
    const cv::Mat &frame)
{
    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(
            input_size_,
            input_size_
        ),
        0,
        0,
        cv::INTER_LINEAR
    );

    // BGR float in [0,1]. BGR->RGB is done for free when
    // the channels are split into the input tensor.
    cv::Mat float_bgr;

    resized.convertTo(
        float_bgr,
        CV_32FC3,
        1.0 / 255.0
    );

    return float_bgr;
}


std::vector<Detection> Predictor::predict(
    const cv::Mat &frame)
{
    std::vector<Detection> detections;

    if (frame.empty())
        return detections;

    cv::Mat input =
        preprocess(frame);

    auto input_tensor =
        infer_request_.get_input_tensor();

    float *input_data =
        input_tensor.data<float>();

    const int H = input.rows;
    const int W = input.cols;

    // Wrap the tensor memory (planar CHW: R, G, B) and let
    // cv::split write straight into it. input is BGR, so
    // plane 0 (B) -> tensor plane 2, plane 2 (R) -> tensor plane 0.
    std::vector<cv::Mat> planes = {
        cv::Mat(H, W, CV_32F, input_data + 2 * H * W),
        cv::Mat(H, W, CV_32F, input_data + 1 * H * W),
        cv::Mat(H, W, CV_32F, input_data)
    };

    cv::split(input, planes);

    infer_request_.infer();

    auto outputs =
        compiled_model_.outputs();

    if (outputs.size() < 2)
    {
        std::cerr
            << "[ERROR] YOLO segmentation model "
               "must have at least 2 outputs."
            << std::endl;

        return detections;
    }

    ov::Tensor detection_output;
    ov::Tensor proto_output;

    /*
     * YOLOv8-seg normally has:
     *
     * output 0:
     * [1, 116, 8400]
     *
     * output 1:
     * [1, 32, 160, 160]
     *
     * But we detect them by dimension.
     */

    for (const auto &output_port : outputs)
    {
        auto tensor =
            infer_request_.get_tensor(
                output_port
            );

        auto shape =
            tensor.get_shape();

        if (shape.size() == 3)
        {
            detection_output =
                tensor;
        }
        else if (shape.size() == 4)
        {
            proto_output =
                tensor;
        }
    }

    if (!detection_output ||
        !proto_output)
    {
        std::cerr
            << "[ERROR] Cannot identify YOLO "
               "segmentation outputs."
            << std::endl;

        return detections;
    }

    return postprocess(
        frame,
        detection_output,
        proto_output
    );
}


std::vector<Detection> Predictor::postprocess(
    const cv::Mat &frame,
    const ov::Tensor &detection_output,
    const ov::Tensor &proto_output)
{
    std::vector<Detection> detections;

    auto shape =
        detection_output.get_shape();

    /*
     * Expected:
     *
     * [1, 116, 8400]
     *
     * 4 box values
     * 80 classes
     * 32 mask coefficients
     */

    if (shape.size() != 3)
        return detections;

    const int channels =
        static_cast<int>(shape[1]);

    const int num_predictions =
        static_cast<int>(shape[2]);

    if (channels < 100)
    {
        std::cerr
            << "[ERROR] Unexpected YOLO output channels: "
            << channels
            << std::endl;

        return detections;
    }

    const int num_classes =
        channels - 4 - 32;

    const float *data =
        detection_output.data<const float>();

    std::vector<cv::Rect> boxes;

    std::vector<float> scores;

    std::vector<int> class_ids;

    std::vector<std::vector<float>>
        mask_coefficients;

    float x_scale =
        static_cast<float>(frame.cols) /
        static_cast<float>(input_size_);

    float y_scale =
        static_cast<float>(frame.rows) /
        static_cast<float>(input_size_);

    /*
     * Best class per prediction. Loop class-major so memory
     * is read sequentially (the tensor layout is [C][N]).
     */

    std::vector<float> best_scores(
        num_predictions,
        0.0f
    );

    std::vector<int> best_classes(
        num_predictions,
        -1
    );

    for (int c = 0; c < num_classes; ++c)
    {
        const float *row =
            data +
            static_cast<size_t>(4 + c) *
            num_predictions;

        for (int i = 0;
             i < num_predictions;
             ++i)
        {
            if (row[i] > best_scores[i])
            {
                best_scores[i] = row[i];
                best_classes[i] = c;
            }
        }
    }

    for (int i = 0;
         i < num_predictions;
         ++i)
    {
        float cx = data[
            0 * num_predictions + i
        ];

        float cy = data[
            1 * num_predictions + i
        ];

        float w = data[
            2 * num_predictions + i
        ];

        float h = data[
            3 * num_predictions + i
        ];

        const float best_score =
            best_scores[i];

        const int best_class =
            best_classes[i];

        if (best_score < confidence_)
            continue;

        float x1 =
            (cx - w * 0.5f) *
            x_scale;

        float y1 =
            (cy - h * 0.5f) *
            y_scale;

        float x2 =
            (cx + w * 0.5f) *
            x_scale;

        float y2 =
            (cy + h * 0.5f) *
            y_scale;

        int ix1 =
            std::max(
                0,
                static_cast<int>(x1)
            );

        int iy1 =
            std::max(
                0,
                static_cast<int>(y1)
            );

        int ix2 =
            std::min(
                frame.cols - 1,
                static_cast<int>(x2)
            );

        int iy2 =
            std::min(
                frame.rows - 1,
                static_cast<int>(y2)
            );

        if (ix2 <= ix1 ||
            iy2 <= iy1)
        {
            continue;
        }

        boxes.emplace_back(
            ix1,
            iy1,
            ix2 - ix1,
            iy2 - iy1
        );

        scores.push_back(
            best_score
        );

        class_ids.push_back(
            best_class
        );

        std::vector<float> coeffs(32);

        for (int m = 0; m < 32; ++m)
        {
            coeffs[m] =
                data[
                    (4 + num_classes + m) *
                    num_predictions +
                    i
                ];
        }

        mask_coefficients.push_back(
            coeffs
        );
    }

    std::vector<int> keep;

    cv::dnn::NMSBoxes(
        boxes,
        scores,
        confidence_,
        0.45f,
        keep
    );

    for (int index : keep)
    {
        Detection detection;

        detection.class_id =
            class_ids[index];

        if (detection.class_id >= 0 &&
            detection.class_id <
            static_cast<int>(
                class_names_.size()
            ))
        {
            detection.class_name =
                class_names_[
                    detection.class_id
                ];
        }
        else
        {
            detection.class_name =
                "class_" +
                std::to_string(
                    detection.class_id
                );
        }

        detection.confidence =
            scores[index];

        detection.box =
            boxes[index];

        detection.mask =
            process_mask(
                mask_coefficients[index],
                proto_output,
                detection.box,
                frame.size()
            );

        detections.push_back(
            detection
        );
    }

    return detections;
}


cv::Mat Predictor::process_mask(
    const std::vector<float> &mask_coefficients,
    const ov::Tensor &proto_output,
    const cv::Rect &box,
    const cv::Size &image_size)
{
    const auto shape =
        proto_output.get_shape();

    if (shape.size() != 4)
        return cv::Mat();

    const int mask_channels =
        static_cast<int>(shape[1]);

    const int mask_height =
        static_cast<int>(shape[2]);

    const int mask_width =
        static_cast<int>(shape[3]);

    if (mask_channels != 32 ||
        mask_coefficients.size() != 32 ||
        image_size.width <= 0 ||
        image_size.height <= 0)
    {
        return cv::Mat();
    }

    const cv::Rect valid_box =
        box &
        cv::Rect(
            0,
            0,
            image_size.width,
            image_size.height
        );

    cv::Mat final_mask =
        cv::Mat::zeros(
            image_size,
            CV_8UC1
        );

    if (valid_box.area() <= 0)
        return final_mask;

    /*
     * logits = coeffs (1x32) * proto (32 x HW)
     * One matrix multiply instead of 32*HW scalar loops.
     * No data is copied: Mats wrap the existing memory.
     */

    const float *proto =
        proto_output.data<const float>();

    const cv::Mat proto_mat(
        mask_channels,
        mask_height * mask_width,
        CV_32F,
        const_cast<float *>(proto)
    );

    const cv::Mat coeffs(
        1,
        mask_channels,
        CV_32F,
        const_cast<float *>(
            mask_coefficients.data()
        )
    );

    cv::Mat logits_flat =
        coeffs * proto_mat;

    const cv::Mat logits =
        logits_flat.reshape(
            1,
            mask_height
        );

    /*
     * Only process the part of the prototype mask
     * that covers the bounding box.
     */

    const float sx =
        static_cast<float>(mask_width) /
        static_cast<float>(image_size.width);

    const float sy =
        static_cast<float>(mask_height) /
        static_cast<float>(image_size.height);

    const int px1 =
        std::max(
            0,
            static_cast<int>(
                std::floor(valid_box.x * sx)
            )
        );

    const int py1 =
        std::max(
            0,
            static_cast<int>(
                std::floor(valid_box.y * sy)
            )
        );

    const int px2 =
        std::min(
            mask_width,
            static_cast<int>(
                std::ceil(
                    (valid_box.x +
                     valid_box.width) * sx
                )
            )
        );

    const int py2 =
        std::min(
            mask_height,
            static_cast<int>(
                std::ceil(
                    (valid_box.y +
                     valid_box.height) * sy
                )
            )
        );

    if (px2 <= px1 ||
        py2 <= py1)
    {
        return final_mask;
    }

    const cv::Mat crop =
        logits(
            cv::Rect(
                px1,
                py1,
                px2 - px1,
                py2 - py1
            )
        );

    // Same region in frame coordinates
    const int fx1 =
        std::clamp(
            static_cast<int>(
                std::lround(px1 / sx)
            ),
            0,
            image_size.width
        );

    const int fy1 =
        std::clamp(
            static_cast<int>(
                std::lround(py1 / sy)
            ),
            0,
            image_size.height
        );

    const int fx2 =
        std::clamp(
            static_cast<int>(
                std::lround(px2 / sx)
            ),
            0,
            image_size.width
        );

    const int fy2 =
        std::clamp(
            static_cast<int>(
                std::lround(py2 / sy)
            ),
            0,
            image_size.height
        );

    const cv::Rect frame_rect(
        fx1,
        fy1,
        fx2 - fx1,
        fy2 - fy1
    );

    if (frame_rect.area() <= 0)
        return final_mask;

    cv::Mat resized;

    cv::resize(
        crop,
        resized,
        frame_rect.size(),
        0,
        0,
        cv::INTER_LINEAR
    );

    // sigmoid(x) > 0.5  <=>  x > 0, so no exp() needed
    cv::Mat binary;

    cv::compare(
        resized,
        0.0f,
        binary,
        cv::CMP_GT
    );

    // Keep the mask only inside the bounding box
    const cv::Rect inter =
        frame_rect & valid_box;

    if (inter.area() > 0)
    {
        binary(
            cv::Rect(
                inter.x - frame_rect.x,
                inter.y - frame_rect.y,
                inter.width,
                inter.height
            )
        ).copyTo(
            final_mask(inter)
        );
    }

    return final_mask;
}


const std::vector<std::string> &
Predictor::get_class_names() const
{
    return class_names_;
}

}