# ROBOCON 2027 – Vision

Computer vision system for the ROBOCON 2027 robot using Intel RealSense D455, YOLOv8 Segmentation, OpenVINO, and ROS 2.

## Overview

This system is responsible for visual perception on the robot. It processes RGB and depth data from the Intel RealSense D455 to perform object segmentation and estimate object distance and 3D position.

The main outputs are:

- Object class
- Confidence score
- Segmentation mask
- Object distance
- Object center
- 3D position `(X, Y, Z)`
- FPS

The processed information is published through ROS 2 and can be used by other robot modules such as navigation and planning.

## System Pipeline

```text
Intel RealSense D455
        │
        ├── RGB Image
        │
        └── Depth Image
                │
                ▼
        Depth-to-Color Alignment
                │
                ▼
        YOLOv8 Segmentation
                │
                ▼
        Object + Segmentation Mask
                │
          ┌─────┴─────┐
          │           │
          ▼           ▼
     Object       Bounding Box
     Distance         Center
          │           │
          │           ▼
          │      3D Position
          │       X, Y, Z
          │           │
          └─────┬─────┘
                │
                ▼
              ROS 2
                │
                ▼
         /vision/objects
                │
                ▼
      Navigation / Planning
```

## Features

- YOLOv8 object segmentation
- Intel RealSense D455 RGB and depth processing
- Depth-to-color alignment
- Distance estimation using segmentation masks
- 3D position estimation
- Real-time FPS calculation
- OpenCV visualization
- ROS 2 integration
- OpenVINO inference
- C++ implementation

## Hardware

- Intel NUC
- Intel RealSense D455
- Robot platform

## Software

- Ubuntu 22.04
- ROS 2
- C++
- OpenCV
- Intel RealSense SDK
- OpenVINO
- YOLOv8 Segmentation

## Configuration

| Parameter | Value |
|---|---|
| Camera | Intel RealSense D455 |
| Color Resolution | 640 × 480 |
| Depth Resolution | 640 × 480 |
| Camera FPS | 30 FPS |
| YOLO Input Size | 640 × 640 |
| Confidence Threshold | 0.7 |
| Inference Device | CPU |
| ROS 2 Topic | `/vision/objects` |

## Model

The system uses YOLOv8n Segmentation converted to OpenVINO IR format.

```text
yolov8n-seg_openvino_model/
├── metadata.yaml
├── yolov8n-seg.xml
└── yolov8n-seg.bin
```

The model is loaded from:

```text
yolov8n-seg.xml
```

The `.xml` file contains the model structure and the `.bin` file contains the model weights.

## Object Segmentation

YOLOv8 Segmentation is used to identify objects and generate segmentation masks.

For each detected object, the system obtains:

```text
Class
Confidence
Bounding Box
Segmentation Mask
```

Example:

```text
Class       : person
Confidence  : 0.91
Bounding Box: (120, 80, 250, 400)
```

The segmentation mask is then used together with the RealSense depth image for distance estimation.

## Depth Processing

The RealSense D455 provides depth information for each pixel.

Before calculating object distance, the depth image is aligned with the color image.

```text
RGB Image
    │
    ▼
YOLO Segmentation
    │
    ▼
Segmentation Mask
    │
    ▼
Aligned Depth Image
    │
    ▼
Depth Values Inside Mask
```

Only valid depth values are used for the distance calculation.

## Distance Estimation

Object distance is calculated from valid depth values inside the segmentation mask.

The processing consists of:

1. Collecting valid depth values inside the mask.
2. Removing invalid depth values.
3. Sorting the depth values.
4. Removing depth outliers using the 10th–90th percentile range.
5. Calculating the median depth.
6. Using the result as the estimated object distance.

Example:

```text
Distance = 1.42 m
```

Using the segmentation mask instead of a single depth pixel helps provide a more stable distance estimation.

## 3D Position

The system calculates the 3D position of each detected object using the center of its bounding box and the RealSense depth data.

The coordinate system is:

```text
X = horizontal position
Y = vertical position
Z = distance from camera
```

Example:

```text
XYZ = (0.15, -0.08, 1.42) m
```

The 3D point is calculated using the RealSense camera intrinsics and depth information.

## FPS

The system calculates the processing FPS in real time.

The FPS value is displayed on the OpenCV output.

Example:

```text
FPS: 28.6
```

The FPS calculation uses smoothing to reduce sudden changes in the displayed value.

## ROS 2

The Vision system runs as a ROS 2 node.

Node name:

```text
yolo_realsense_vision
```

Object information is published through:

```text
/vision/objects
```

Message type:

```text
std_msgs/msg/String
```

The message contains JSON-formatted object information.

Example:

```json
[
  {
    "class": "person",
    "confidence": 0.91,
    "center_x": 320,
    "center_y": 240,
    "distance": 1.42,
    "x": 0.15,
    "y": -0.08,
    "z": 1.42
  }
]
```

If depth information is unavailable:

```json
[
  {
    "class": "person",
    "confidence": 0.91,
    "center_x": 320,
    "center_y": 240,
    "distance": null,
    "x": null,
    "y": null,
    "z": null
  }
]
```

The topic can be subscribed to by other robot modules.

## Visualization

The OpenCV window displays:

- Bounding box
- Segmentation mask
- Object class
- Confidence
- Object distance
- 3D position
- FPS

Example:

```text
person 0.91
Distance: 1.42 m
XYZ: (0.15, -0.08, 1.42) m
FPS: 28.6
```

Press `Q` to stop the node.

## Project Structure

```text
computer_vision/
│
├── src/
│   └── vision_cpp/
│       │
│       ├── CMakeLists.txt
│       ├── package.xml
│       │
│       ├── include/
│       │   └── vision_cpp/
│       │       ├── predictor.hpp
│       │       └── utils.hpp
│       │
│       └── src/
│           ├── predictor.cpp
│           ├── utils.cpp
│           └── vision.cpp
│
└── yolov8n-seg_openvino_model/
    ├── metadata.yaml
    ├── yolov8n-seg.xml
    └── yolov8n-seg.bin
```

## Source Files

### `vision.cpp`

Main ROS 2 node responsible for:

- Initializing ROS 2
- Initializing the RealSense D455
- Capturing RGB frames
- Capturing depth frames
- Aligning depth to color
- Running YOLO segmentation
- Calculating object distance
- Calculating 3D position
- Publishing object information
- Displaying visualization
- Calculating FPS

### `predictor.cpp`

Handles the YOLO inference process:

- Loading the OpenVINO model
- Image preprocessing
- Running inference
- Processing YOLO outputs
- Generating object detections
- Generating segmentation masks

### `predictor.hpp`

Defines the `Predictor` class and `Detection` structure.

The detection data contains:

```text
class_id
class_name
confidence
box
mask
distance
point_3d
has_depth
```

### `utils.cpp`

Contains utility functions for:

- FPS calculation
- Depth processing
- Mask-based distance calculation
- 3D point calculation
- Detection visualization
- FPS visualization
- Depth information visualization

### `utils.hpp`

Contains the declarations for the utility functions and `FPSCalculator`.

## Build

Source ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
```

Go to the workspace:

```bash
cd ~/computer_vision
```

Build the package:

```bash
colcon build \
    --symlink-install \
    --packages-select vision_cpp \
    --cmake-args \
    -DOpenVINO_DIR=/home/barelangv/.local/lib/python3.10/site-packages/openvino/cmake
```

After building:

```bash
source install/setup.bash
```

## Run

Run the Vision node:

```bash
ros2 run vision_cpp vision
```

The default model should point to:

```text
/home/barelangv/computer_vision/src/vision_cpp/yolov8n-seg_openvino_model/yolov8n-seg.xml
```

The model can also be specified manually:

```bash
ros2 run vision_cpp vision --ros-args \
    -p model:=/home/barelangv/computer_vision/src/vision_cppyolov8n-seg_openvino_model/yolov8n-seg.xml
```

Set the confidence threshold:

```bash
ros2 run vision_cpp vision --ros-args \
    -p confidence:=0.7
```

Set the inference device:

```bash
ros2 run vision_cpp vision --ros-args \
    -p device:=CPU
```

## ROS 2 Topic

Check the published topic:

```bash
ros2 topic list
```

Check the Vision topic:

```bash
ros2 topic echo /vision/objects
```

Check the topic type:

```bash
ros2 topic type /vision/objects
```

Expected output:

```text
std_msgs/msg/String
```

## Data Flow

The complete processing flow is:

```text
RealSense D455
      │
      ▼
RGB + Depth
      │
      ▼
Depth Alignment
      │
      ▼
YOLOv8 Segmentation
      │
      ▼
Object Detection
      │
      ├── Class
      ├── Confidence
      ├── Bounding Box
      └── Segmentation Mask
                  │
                  ▼
             Depth Mask
                  │
            ┌─────┴─────┐
            ▼           ▼
        Distance     3D Position
                       X Y Z
            │           │
            └─────┬─────┘
                  ▼
                ROS 2
                  │
                  ▼
           /vision/objects
```

## Role in Robot System

The Vision system functions as the perception module of the robot.

Its responsibility is to convert camera data into information that can be processed by other robot systems.

```text
Camera
   │
   ▼
Vision
   │
   ├── Object
   ├── Distance
   ├── Segmentation
   └── 3D Position
           │
           ▼
      Robot System
```

The output can be used for further processing such as:

- Navigation
- Object localization
- Path planning
- Autonomous decision making

## Development

Current development focuses on integrating object segmentation and depth-based perception with the robot system.

Possible future development:

- Custom object classes
- Custom dataset
- Object tracking
- Improved 3D localization
- Navigation integration
- Path planning integration
- Custom ROS 2 message
- Performance optimization on Intel NUC
