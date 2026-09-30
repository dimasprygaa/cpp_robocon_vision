#pragma once

#include "vision_cpp/predictor.hpp"   // Detection

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include <chrono>
#include <string>

namespace vision_cpp
{

// ============================================================
// FPS CALCULATOR (exponential moving average)
// ============================================================

class FPSCalculator
{
public:

    // alpha = weight of the previous value (0..1). Higher = smoother.
    explicit FPSCalculator(double alpha = 0.9);

    // Call once per frame. Returns smoothed FPS.
    double update();

private:

    // NOTE: order matches the initializer list in utils.cpp
    double alpha_;
    double fps_;
    bool initialized_;
    std::chrono::steady_clock::time_point last_time_;
};

// ============================================================
// DEPTH
// ============================================================

// Median distance (meters) of the pixels inside the mask.
// Returns -1.0f if not available.
float get_mask_distance(
    const rs2::depth_frame &depth_frame,
    const cv::Mat &mask
);

bool get_3d_point(
    const rs2::depth_frame &depth,
    int x,
    int y,
    cv::Point3f &point
);

// ============================================================
// DRAWING
// ============================================================

void draw_detection(
    cv::Mat &frame,
    const Detection &detection
);

void draw_fps(
    cv::Mat &frame,
    double fps
);

void draw_depth_info(
    cv::Mat &frame,
    const Detection &detection
);

}  // namespace vision_cpp
