cmake_minimum_required(VERSION 3.25)
project(onnx-rt-test)

if(NOT DEFINED ONNX_RT_ROOT)
    set(ONNX_RT_ROOT ${CMAKE_CURRENT_LIST_DIR}/onnxruntime-linux-x64-1.14.1)
endif()

include_directories(${ONNX_RT_ROOT}/include)
link_directories(${ONNX_RT_ROOT}/lib)
link_libraries(onnxruntime)

add_executable(test test.cpp)
