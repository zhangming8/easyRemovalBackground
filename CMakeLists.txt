cmake_minimum_required(VERSION 3.16)
project(AITOOLS)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(thirdparty)
add_subdirectory(src)

message("[${PROJECT_NAME}] CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message("[${PROJECT_NAME}] CUDA_TOOLKIT_INCLUDE ${CUDA_TOOLKIT_INCLUDE}")
message("[${PROJECT_NAME}] CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}")
message("[${PROJECT_NAME}] CUDA_LIBRARIES ${CUDA_LIBRARIES}")
message("[${PROJECT_NAME}] APP_LIBRARIES ${APP_LIBRARIES}")
message("[${PROJECT_NAME}] SPDLOG_INCLUDE_DIR ${SPDLOG_INCLUDE_DIR}")

set(Qt5_DIR "C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/lib/cmake/Qt5")
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets Multimedia MultimediaWidgets)
message("[${PROJECT_NAME}] Qt5Core_INCLUDE_DIRS: ${Qt5Core_INCLUDE_DIRS}")
message("[${PROJECT_NAME}] Qt5Core_VERSION: ${Qt5Core_VERSION}")
message("[${PROJECT_NAME}] Qt5Core_DIR: ${Qt5Core_DIR}")
message("[${PROJECT_NAME}] Qt5Gui_DIR: ${Qt5Gui_DIR}")
message("[${PROJECT_NAME}] Qt5Widgets_DIR: ${Qt5Widgets_DIR}")
message("[${PROJECT_NAME}] Qt5Multimedia_DIR: ${Qt5Multimedia_DIR}")

find_package(OpenCV REQUIRED)
message("[${PROJECT_NAME}] OpenCV found: ${OpenCV_FOUND}")
message("[${PROJECT_NAME}] OpenCV verion: ${OpenCV_VERSION}")
message("[${PROJECT_NAME}] OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message("[${PROJECT_NAME}] OpenCV libraries: ${OpenCV_LIBS}")
message("[${PROJECT_NAME}] OpenCV_DIR ${OpenCV_DIR}")

# get_cmake_property(_variableNames VARIABLES)
# list(SORT _variableNames)
# foreach(_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
#     if(_variableName MATCHES "Qt5")
#         message(STATUS "${_variableName}=${${_variableName}}")
#     endif()
# endforeach()

include_directories(
    ${CUDA_TOOLKIT_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(${ONNXRUNTIME_LIB_DIR})

set(LIBS
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}
    ${ONNXRUNTIME_LIBS}
    Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Multimedia Qt5::MultimediaWidgets
    ${APP_LIBRARIES}
)


qt5_wrap_cpp(moc_source_segment "include/interface/segment/mainwindow.h")

add_executable(easyRemovalBackground WIN32 main_segment.cpp "src/interface/segment/mainwindow.cpp" ${moc_source_segment})    
target_link_libraries(easyRemovalBackground ${LIBS})

add_executable(test main_segment.cpp "src/interface/segment/mainwindow.cpp" ${moc_source_segment})
target_link_libraries(test ${LIBS})

message(STATUS "===================>>>>>>>>> Build configuration complete <<<<<<<<===================")
