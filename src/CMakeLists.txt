
cmake_minimum_required(VERSION 3.18)
project(app_lib VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)

set(project_src
    app.cpp
    utils/util.cpp
    utils/logger.cpp
    algo/detect/onnx_base.cpp
    algo/matting/inference_mat.cpp
)

IF (CMAKE_BUILD_TYPE STREQUAL Debug)
  ADD_DEFINITIONS(-DDEBUG)
ENDIF()

message("[${PROJECT_NAME}] OpenCV_DIR ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
message("[${PROJECT_NAME}] OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS}")
# message("[${PROJECT_NAME}] OpenCV_LIBS ${OpenCV_LIBS}")
message("[${PROJECT_NAME}] OpenCV_VERSION ${OpenCV_VERSION}")

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDI GPU Computing Toolkit/CUDA/v12.4" CACHE INTERNAL "")
message("[${PROJECT_NAME}] CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR}")
find_package(CUDA REQUIRED)
message("[${PROJECT_NAME}] CUDA_TOOLKIT_INCLUDE ${CUDA_TOOLKIT_INCLUDE}")
message("[${PROJECT_NAME}] CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}")

set(ONNXRUNTIME_DIR "C:/project/onnxruntime-win-x64-gpu-1.23.2" CACHE INTERNAL "")
set(ONNXRUNTIME_LIBS "onnxruntime_providers_cuda;onnxruntime_providers_shared;onnxruntime_providers_tensorrt;onnxruntime;" CACHE INTERNAL "")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include" CACHE INTERNAL "")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib" CACHE INTERNAL "")
message("[${PROJECT_NAME}] ONNXRUNTIME_DIR ${ONNXRUNTIME_DIR}")

message("[${PROJECT_NAME}] SPDLOG_INCLUDE_DIR ${SPDLOG_INCLUDE_DIR}")

include_directories(
    ${SPDLOG_INCLUDE_DIR}
    ${CUDA_TOOLKIT_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

link_directories(${ONNXRUNTIME_LIB_DIR})

set(LIBS
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}
    ${ONNXRUNTIME_LIBS}
)

add_library(${PROJECT_NAME} ${project_src})
target_link_libraries(${PROJECT_NAME} ${LIBS})
set(APP_LIBRARIES ${PROJECT_NAME} CACHE INTERNAL "")
