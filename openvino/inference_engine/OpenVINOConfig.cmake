get_filename_component(OpenVINO_CONFIG_PATH "${CMAKE_CURRENT_LIST_DIR}" REALPATH)

find_library(OpenVINO_IE NAMES inference_engine.lib PATHS "${OpenVINO_CONFIG_PATH}/lib" )
find_library(OpenVINO_CPU NAMES cpu_extension.lib PATHS "${OpenVINO_CONFIG_PATH}/lib" )
set(OpenVINO_INCLUDE_DIRS "${OpenVINO_CONFIG_PATH}/include")
set(OpenVINO_LIBS ${OpenVINO_IE} ${OpenVINO_CPU})
mark_as_advanced(OpenVINO_INCLUDE_DIRS OpenVINO_LIBS )
set(OpenVINO_FOUND TRUE)
