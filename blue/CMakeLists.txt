project(yuv-blur)
#cmake_minimum_required(VERSION 2.8)

#set(OpenCV_DIR D:/users/think/opencv/opencv/sources/OpenCV3.3_VS2013_x64/install)
#find_package(OpenCV)
#include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(OpenCL)
#include_directories(${OpenCL_INCLUDE_DIR})

file(GLOB src_files *.cpp *.h)
add_executable(yuv-blur ${src_files})
#target_link_libraries(yuv-blur ${OpenCV_LIBS} ${OpenCL_LIBRARY})
target_link_libraries(yuv-blur ${OpenCV_LIBS})
