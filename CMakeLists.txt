# CMake 最低版本号要求
cmake_minimum_required(VERSION 2.8)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()

#默认由环境变量 OpenCV_DIR 定位OpenCVConfig.cmake
find_package( OpenCV REQUIRED )

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

#添加要编译的子目录
add_subdirectory(demo)
add_subdirectory(blue)