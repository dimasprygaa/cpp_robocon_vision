cmake_minimum_required(VERSION 3.8)

project(vision_cpp)

# Without this, colcon builds with NO optimization (-O0) which makes the
# pre/post-processing loops many times slower.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================
# ROS 2
# ============================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# ============================================================
# OpenCV
# ============================================================
find_package(OpenCV REQUIRED)

# ============================================================
# RealSense
# ============================================================
find_package(realsense2 REQUIRED)

# ============================================================
# OpenVINO
# ============================================================
set(OpenVINO_DIR
    "/home/barelangv/.local/lib/python3.10/site-packages/openvino/cmake"
)

find_package(OpenVINO REQUIRED)

# ============================================================
# SOURCE
# ============================================================
add_executable(vision
  src/vision.cpp
  src/predictor.cpp
  src/utils.cpp
)

# ============================================================
# INCLUDE DIRECTORIES
# ============================================================
target_include_directories(vision
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# ROS DEPENDENCIES
# ============================================================
ament_target_dependencies(vision
  rclcpp
  std_msgs
)

# ============================================================
# LIBRARIES
# ============================================================
target_link_libraries(vision
  ${OpenCV_LIBS}
  realsense2
  openvino::runtime
)

# ============================================================
# INSTALL
# ============================================================
install(
  TARGETS vision
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_package()