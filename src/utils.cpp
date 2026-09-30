#include "vision_cpp/utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <vector>

namespace vision_cpp
{

// ============================================================
// FPS CALCULATOR
// ============================================================

FPSCalculator::FPSCalculator(double alpha)
    : alpha_(alpha),
      fps_(0.0),
      initialized_(false),
      last_time_(std::chrono::steady_clock::now())
{
}

double FPSCalculator::update()
{
    const auto current_time =
        std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed =
        current_time - last_time_;

    last_time_ = current_time;

    const double dt = elapsed.count();

    if (dt <= 0.0)
    {
        return fps_;
    }

    const double instant_fps = 1.0 / dt;

    if (!initialized_)
    {
        fps_ = instant_fps;
        initialized_ = true;
    }
    else
    {
        fps_ =
            alpha_ * fps_ +
            (1.0 - alpha_) * instant_fps;
    }

    return fps_;
}


// ============================================================
// GET DISTANCE FROM SEGMENTATION MASK
// ============================================================

float get_mask_distance(
    const rs2::depth_frame &depth_frame,
    const cv::Mat &mask)
{
    if (!depth_frame)
    {
        return -1.0f;
    }

    if (mask.empty())
    {
        return -1.0f;
    }

    const int width =
        depth_frame.get_width();

    const int height =
        depth_frame.get_height();

    if (mask.cols != width ||
        mask.rows != height)
    {
        return -1.0f;
    }

    // Read the raw Z16 buffer directly. depth_frame.get_distance()
    // goes through the C API for every pixel, which is very slow.
    const uint8_t *base =
        static_cast<const uint8_t *>(
            depth_frame.get_data()
        );

    const int stride =
        depth_frame.get_stride_in_bytes();

    const float units =
        depth_frame.get_units();

    // Sample every 2nd pixel: 4x fewer reads, same median.
    constexpr int STEP = 2;

    std::vector<float> distances;

    distances.reserve(
        static_cast<size_t>(
            width * height / (STEP * STEP * 4)
        )
    );

    for (int y = 0; y < height; y += STEP)
    {
        const uchar *mask_row =
            mask.ptr<uchar>(y);

        const uint16_t *depth_row =
            reinterpret_cast<const uint16_t *>(
                base +
                static_cast<size_t>(y) * stride
            );

        for (int x = 0; x < width; x += STEP)
        {
            if (mask_row[x] == 0)
            {
                continue;
            }

            const uint16_t raw =
                depth_row[x];

            if (raw == 0)
            {
                continue;
            }

            distances.push_back(
                static_cast<float>(raw) * units
            );
        }
    }

    if (distances.empty())
    {
        return -1.0f;
    }

    // --------------------------------------------------------
    // Sort
    // --------------------------------------------------------

    std::sort(
        distances.begin(),
        distances.end()
    );

    // --------------------------------------------------------
    // Remove outliers
    // 10th - 90th percentile
    // --------------------------------------------------------

    size_t lower_index =
        static_cast<size_t>(
            distances.size() * 0.10
        );

    size_t upper_index =
        static_cast<size_t>(
            distances.size() * 0.90
        );

    if (upper_index >= distances.size())
    {
        upper_index =
            distances.size() - 1;
    }

    if (lower_index > upper_index)
    {
        lower_index = 0;
        upper_index =
            distances.size() - 1;
    }

    std::vector<float> filtered;

    filtered.reserve(
        upper_index - lower_index + 1
    );

    for (size_t i = lower_index;
         i <= upper_index;
         ++i)
    {
        filtered.push_back(distances[i]);
    }

    if (filtered.empty())
    {
        return -1.0f;
    }

    // --------------------------------------------------------
    // Median
    // --------------------------------------------------------

    const size_t middle =
        filtered.size() / 2;

    if (filtered.size() % 2 == 0)
    {
        return (
            filtered[middle - 1] +
            filtered[middle]
        ) * 0.5f;
    }

    return filtered[middle];
}


// ============================================================
// GET 3D POINT
// ============================================================

bool get_3d_point(
    const rs2::depth_frame &depth,
    int x,
    int y,
    cv::Point3f &point)
{
    if (!depth)
    {
        return false;
    }

    // --------------------------------------------------------
    // Validate pixel
    // --------------------------------------------------------

    if (x < 0 ||
        x >= depth.get_width() ||
        y < 0 ||
        y >= depth.get_height())
    {
        return false;
    }

    // --------------------------------------------------------
    // Get depth
    // --------------------------------------------------------

    const float depth_value =
        depth.get_distance(x, y);

    if (depth_value <= 0.0f)
    {
        return false;
    }

    if (!std::isfinite(depth_value))
    {
        return false;
    }

    // --------------------------------------------------------
    // Get camera intrinsics
    // --------------------------------------------------------

    const rs2::video_stream_profile video_profile =
        depth.get_profile()
            .as<rs2::video_stream_profile>();

    const rs2_intrinsics intrinsics =
        video_profile.get_intrinsics();

    // --------------------------------------------------------
    // Pixel
    // --------------------------------------------------------

    float pixel[2] =
    {
        static_cast<float>(x),
        static_cast<float>(y)
    };

    // --------------------------------------------------------
    // XYZ
    // --------------------------------------------------------

    float xyz[3];

    rs2_deproject_pixel_to_point(
        xyz,
        &intrinsics,
        pixel,
        depth_value
    );

    // --------------------------------------------------------
    // OpenCV point

    point.x = xyz[0];
    point.y = xyz[1];
    point.z = xyz[2];

    return true;
}


// DRAW DETECTION

void draw_detection(
    cv::Mat &frame,
    const Detection &detection)
{
    if (frame.empty())
    {
        return;
    }

    // Bounding box

    cv::Rect box =
        detection.box;

    box &= cv::Rect(
        0,
        0,
        frame.cols,
        frame.rows
    );

    if (box.width <= 0 ||
        box.height <= 0)
    {
        return;
    }

    cv::rectangle(
        frame,
        box,
        cv::Scalar(0, 255, 0),
        2
    );

    // Segmentation mask

    if (!detection.mask.empty())
    {
        cv::Mat mask = detection.mask;

        if (mask.size() != frame.size())
        {
            cv::resize(
                mask,
                mask,
                frame.size(),
                0,
                0,
                cv::INTER_NEAREST
            );
        }

        // Blend only inside the bounding box
        cv::Mat roi = frame(box);

        const cv::Mat green(
            roi.size(),
            roi.type(),
            cv::Scalar(0, 255, 0)
        );

        cv::Mat blended;

        cv::addWeighted(
            roi,
            0.65,
            green,
            0.35,
            0.0,
            blended
        );

        blended.copyTo(
            roi,
            mask(box)
        );
    }

    // Label

    std::ostringstream label;

    label
        << detection.class_name
        << " "
        << std::fixed
        << std::setprecision(2)
        << detection.confidence;

    if (detection.distance > 0.0f)
    {
        label
            << " | "
            << std::fixed
            << std::setprecision(2)
            << detection.distance
            << " m";
    }

    if (detection.has_depth)
    {
        label
            << " | XYZ "
            << std::fixed
            << std::setprecision(2)
            << detection.point_3d.x
            << ","
            << detection.point_3d.y
            << ","
            << detection.point_3d.z;
    }

    // Text size

    int baseline = 0;

    const cv::Size text_size =
        cv::getTextSize(
            label.str(),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
            &baseline
        );

    const int text_x =
        box.x;

    const int text_y =
        std::max(
            box.y - 5,
            text_size.height + 5
        );

    // --------------------------------------------------------
    // Background
    // --------------------------------------------------------

    cv::rectangle(
        frame,
        cv::Point(
            text_x,
            text_y - text_size.height - 5
        ),
        cv::Point(
            text_x + text_size.width + 5,
            text_y + baseline
        ),
        cv::Scalar(0, 255, 0),
        cv::FILLED
    );

    // Text


    cv::putText(
        frame,
        label.str(),
        cv::Point(
            text_x + 2,
            text_y
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(0, 0, 0),
        1,
        cv::LINE_AA
    );
}


// DRAW FPS

void draw_fps(
    cv::Mat &frame,
    double fps)
{
    if (frame.empty())
    {
        return;
    }

    std::ostringstream text;

    text
        << "FPS: "
        << std::fixed
        << std::setprecision(1)
        << fps;

    cv::putText(
        frame,
        text.str(),
        cv::Point(20, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA
    );
}



// DRAW DEPTH / XYZ

void draw_depth_info(
    cv::Mat &frame,
    const Detection &detection)
{
    if (frame.empty())
    {
        return;
    }

    if (!detection.has_depth)
    {
        return;
    }

    const cv::Point center(
        detection.box.x +
            detection.box.width / 2,

        detection.box.y +
            detection.box.height / 2
    );

    if (center.x < 0 ||
        center.x >= frame.cols ||
        center.y < 0 ||
        center.y >= frame.rows)
    {
        return;
    }

    // --------------------------------------------------------
    // Center point
    // --------------------------------------------------------

    cv::circle(
        frame,
        center,
        5,
        cv::Scalar(0, 0, 255),
        -1
    );

    // --------------------------------------------------------
    // XYZ text
    // --------------------------------------------------------

    std::ostringstream xyz_text;

    xyz_text
        << "X: "
        << std::fixed
        << std::setprecision(2)
        << detection.point_3d.x
        << "m  Y: "
        << detection.point_3d.y
        << "m  Z: "
        << detection.point_3d.z
        << "m";

    int text_y =
        center.y + 25;

    if (text_y >= frame.rows)
    {
        text_y =
            frame.rows - 10;
    }

    cv::putText(
        frame,
        xyz_text.str(),
        cv::Point(
            std::max(center.x - 100, 0),
            text_y
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
    );
}

} // namespace vision_cpp