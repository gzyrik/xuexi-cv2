# 项目信息
project( face_detection )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 指定生成目标
add_executable( face_detection face_detection.cpp )
target_link_libraries( face_detection ${OpenCV_LIBS} )

# 项目信息
project( simple_svm )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 指定生成目标
add_executable( simple_svm simple_svm.cpp )
target_link_libraries( simple_svm ${OpenCV_LIBS} )

# 项目信息
project( vb )
include_directories( ${OpenCV_INCLUDE_DIRS} ${OpenVINO_INCLUDE_DIRS})
# 指定生成目标
add_executable( vb vb.cpp )
target_link_libraries( vb ${OpenCV_LIBS} ${OpenVINO_LIBS})

# 项目信息
project( nonlinear_svm )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 指定生成目标
add_executable( nonlinear_svm nonlinear_svm.cpp )
target_link_libraries( nonlinear_svm ${OpenCV_LIBS} )

# 项目信息
project( simple_pca )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 指定生成目标
add_executable( simple_pca simple_pca.cpp )
target_link_libraries( simple_pca ${OpenCV_LIBS} )

# 项目信息
project( caffe_dnn )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 指定生成目标
add_executable( caffe_dnn caffe_dnn.cpp )
target_link_libraries( caffe_dnn ${OpenCV_LIBS} )


# 项目信息
project( stitch_pano )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 指定生成目标
add_executable( stitch_pano stitch_pano.cpp )
target_link_libraries( stitch_pano ${OpenCV_LIBS} )

# 项目信息
project( yuview )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 指定生成目标
add_executable( yuview yuview.cpp )
target_link_libraries( yuview ${OpenCV_LIBS} )

# 项目信息
project( robust_pca )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 指定生成目标
add_executable( robust_pca robust_pca.cpp )
target_link_libraries( robust_pca ${OpenCV_LIBS} )
