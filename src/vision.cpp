#include "vision_cpp/predictor.hpp"
#include "vision_cpp/utils.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <librealsense2/rs.hpp>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>


class VisionNode : public rclcpp::Node
{
public:

    VisionNode()
        : Node("yolo_realsense_vision")
    {
        RCLCPP_INFO(
            get_logger(),
            "Starting YOLO + RealSense C++ node"
        );

        /*
         * PARAMETERS
         */

        model_path_ =
            declare_parameter<std::string>(
                "model",
                "/home/barelangv/computer_vision/srv/vision_cpp/yolo11n-seg_openvino_model/yolo11n-seg.xml"
            );

        device_ =
            declare_parameter<std::string>(
                "device",
                "CPU"
            );

        confidence_ =
            declare_parameter<double>(
                "confidence",
                0.7
            );

        /*
         * PREDICTOR
         */

        RCLCPP_INFO(
            get_logger(),
            "Loading predictor..."
        );

        predictor_ =
            std::make_unique<
                vision_cpp::Predictor
            >(
                model_path_,
                device_,
                static_cast<float>(
                    confidence_
                ),
                640
            );

        /*
         * REALSENSE
         */

        RCLCPP_INFO(
            get_logger(),
            "Starting RealSense D455..."
        );

        config_.enable_stream(
            RS2_STREAM_COLOR,
            640,
            480,
            RS2_FORMAT_BGR8,
            30
        );

        config_.enable_stream(
            RS2_STREAM_DEPTH,
            640,
            480,
            RS2_FORMAT_Z16,
            30
        );

        profile_ =
            pipeline_.start(
                config_
            );

        depth_scale_ =
            profile_
            .get_device()
            .first<rs2::depth_sensor>()
            .get_depth_scale();

        RCLCPP_INFO(
            get_logger(),
            "Depth scale: %f",
            depth_scale_
        );

        /*
         * ALIGN
         */

        align_ =
            std::make_unique<
                rs2::align
            >(
                RS2_STREAM_COLOR
            );

        /*
         * ROS PUBLISHER
         */

        object_pub_ =
            create_publisher<
                std_msgs::msg::String
            >(
                "/vision/objects",
                10
            );

        /*
         * TIMER
         */

        timer_ =
            create_wall_timer(
                std::chrono::milliseconds(1),
                std::bind(
                    &VisionNode::process,
                    this
                )
            );

        RCLCPP_INFO(
            get_logger(),
            "Vision node started"
        );

        RCLCPP_INFO(
            get_logger(),
            "RealSense: 640x480 @ 30 FPS"
        );

        RCLCPP_INFO(
            get_logger(),
            "OpenVINO device: %s",
            device_.c_str()
        );
    }


    ~VisionNode()
    {
        try
        {
            pipeline_.stop();
        }
        catch (...)
        {
        }

        cv::destroyAllWindows();
    }


private:

    void process()
    {
        const auto t_start =
            std::chrono::steady_clock::now();

        rs2::frameset frames;

        try
        {
            frames =
                pipeline_.wait_for_frames(
                    100
                );
        }
        catch (
            const rs2::error &e
        )
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                5000,
                "RealSense: %s",
                e.what()
            );

            return;
        }

        /*
         * ALIGN DEPTH -> COLOR
         */

        frames =
            align_->process(
                frames
            );

        rs2::video_frame color_frame =
            frames.get_color_frame();

        rs2::depth_frame depth_frame =
            frames.get_depth_frame();

        if (!color_frame ||
            !depth_frame)
        {
            return;
        }

        /*
         * COLOR -> OpenCV
         */

        cv::Mat frame(
            cv::Size(
                color_frame.get_width(),
                color_frame.get_height()
            ),
            CV_8UC3,
            (void *)color_frame.get_data(),
            cv::Mat::AUTO_STEP
        );

        /*
         * YOLO
         */

        const auto t_predict_begin =
            std::chrono::steady_clock::now();

        auto detections =
            predictor_->predict(
                frame
            );

        const auto t_predict_end =
            std::chrono::steady_clock::now();

        /*
         * OUTPUT IMAGE
         */

        cv::Mat annotated =
            frame.clone();

        /*
         * ROS DATA
         */

        std::ostringstream json;

        json << "[";

        bool first_object = true;

        /*
         * PROCESS OBJECTS
         */

        for (auto &detection :
             detections)
        {
            /*
             * DEPTH FROM MASK
             */

            if (!detection.mask.empty())
            {
                detection.distance =
                    vision_cpp::
                    get_mask_distance(
                        depth_frame,
                        detection.mask
                    );
            }

            /*
             * CENTER
             */

            int cx =
                detection.box.x +
                detection.box.width / 2;

            int cy =
                detection.box.y +
                detection.box.height / 2;

            /*
             * 3D POINT
             *
             * Since depth frame is aligned
             * to color, center coordinates
             * can be used directly.
             */

            cv::Point3f point;

            bool valid_3d =
                vision_cpp::
                get_3d_point(
                    depth_frame,
                    cx,
                    cy,
                    point
                );

            detection.point_3d =
                point;

            detection.has_depth =
                valid_3d;

            /*
             * DRAW BOX / TEXT
             */

            vision_cpp::
            draw_detection(
                annotated,
                detection
            );

            vision_cpp::
            draw_depth_info(
                annotated,
                detection
            );

            /*
             * PRINT TERMINAL
             */

            std::cout
                << detection.class_name
                << " | "
                << std::fixed
                << std::setprecision(2)
                << detection.confidence
                << " | Center=("
                << cx
                << ","
                << cy
                << ")";

            if (detection.distance > 0)
            {
                std::cout
                    << " | Distance="
                    << std::setprecision(2)
                    << detection.distance
                    << " m";
            }
            else
            {
                std::cout
                    << " | Distance=N/A";
            }

            if (valid_3d)
            {
                std::cout
                    << " | XYZ=("
                    << point.x
                    << ", "
                    << point.y
                    << ", "
                    << point.z
                    << ") m";
            }

            std::cout
                << std::endl;

            /*
             * JSON-LIKE ROS DATA
             */

            if (!first_object)
                json << ",";

            first_object = false;

            json
                << "{"
                << "\"class\":\""
                << detection.class_name
                << "\","

                << "\"confidence\":"
                << detection.confidence
                << ","

                << "\"center_x\":"
                << cx
                << ","

                << "\"center_y\":"
                << cy
                << ",";

            if (detection.distance > 0)
            {
                json
                    << "\"distance\":"
                    << detection.distance
                    << ",";
            }
            else
            {
                json
                    << "\"distance\":null,";
            }

            if (valid_3d)
            {
                json
                    << "\"x\":"
                    << point.x
                    << ","

                    << "\"y\":"
                    << point.y
                    << ","

                    << "\"z\":"
                    << point.z;
            }
            else
            {
                json
                    << "\"x\":null,"
                    << "\"y\":null,"
                    << "\"z\":null";
            }

            json << "}";
        }

        json << "]";

        /*
         * PUBLISH ROS
         */

        std_msgs::msg::String msg;

        msg.data =
            json.str();

        object_pub_->publish(
            msg
        );

        /*
         * FPS
         */

        vision_cpp::draw_fps(
            annotated,
            fps_calculator_.update()
        );

        {
            using ms = std::chrono::duration<double, std::milli>;

            const auto t_end =
                std::chrono::steady_clock::now();

            RCLCPP_INFO_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "wait+align+copy: %.1f ms | predict: %.1f ms | depth+draw+publish: %.1f ms",
                ms(t_predict_begin - t_start).count(),
                ms(t_predict_end - t_predict_begin).count(),
                ms(t_end - t_predict_end).count()
            );
        }

        /*
         * DISPLAY
         */

        cv::imshow(
            "YOLOv8 Seg + RealSense D455",
            annotated
        );

        int key =
            cv::waitKey(1);

        if (key == 'q' ||
            key == 'Q')
        {
            rclcpp::shutdown();
        }
    }


    /*
     * ROS
     */

    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Publisher<
        std_msgs::msg::String
    >::SharedPtr object_pub_;


    /*
     * REALSENSE
     */

    rs2::pipeline pipeline_;

    rs2::config config_;

    rs2::pipeline_profile profile_;

    std::unique_ptr<
        rs2::align
    > align_;

    float depth_scale_;

    vision_cpp::FPSCalculator fps_calculator_;


    /*
     * YOLO
     */

    std::unique_ptr<
        vision_cpp::Predictor
    > predictor_;

    std::string model_path_;

    std::string device_;

    double confidence_;
};


int main(
    int argc,
    char *argv[])
{
    rclcpp::init(
        argc,
        argv
    );

    try
    {
        auto node =
            std::make_shared<
                VisionNode
            >();

        rclcpp::spin(
            node
        );
    }
    catch (
        const std::exception &e
    )
    {
        std::cerr
            << "[ERROR] "
            << e.what()
            << std::endl;
    }

    rclcpp::shutdown();

    return 0;
}