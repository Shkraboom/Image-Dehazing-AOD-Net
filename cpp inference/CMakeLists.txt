cmake_minimum_required(VERSION 3.10)
project(ImageDehazing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(/usr/include/x86_64-linux-gnu)
include_directories(/usr/local/cuda-12.3/include)

link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(/usr/local/cuda-12.3/lib64)

add_executable(dehaze_program main.cpp)

target_link_libraries(dehaze_program ${OpenCV_LIBS} nvinfer nvonnxparser cudart)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/cuda-12.3/include
)

target_link_libraries(dehaze_program
        ${OpenCV_LIBS}
        nvinfer
        nvonnxparser
        cudart
        ${CUDAToolkit_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/libnvinfer.so
        /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
)