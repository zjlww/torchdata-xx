cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(torchdata-xx
    VERSION 1.0
    DESCRIPTION "A PyTorch dataloader in C++ with Python binding."
    LANGUAGES C CXX
)
include(FetchContent)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

set(CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib")

# Disallow In-source Building
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed")
endif()

find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include(cmake/torch.cmake)
find_package (TBB REQUIRED)
find_package (Boost REQUIRED COMPONENTS thread system)

file(GLOB SRC_CPP "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp")
file(GLOB SRC_CPP_TEXT "${CMAKE_CURRENT_SOURCE_DIR}/csrc/text/*.cpp")

add_library(SHAREDEP INTERFACE)
target_link_libraries(SHAREDEP INTERFACE Torch Boost::thread Boost::system tbb sox soxr)
target_include_directories(SHAREDEP INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/")

# A multi-threading version of ESpeak-NG is absorbed.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/espeak-ng)

# Python Extension:
pybind11_add_module(torchdataxx_C binding.cpp ${SRC_CPP} ${SRC_CPP_TEXT})
target_link_libraries(torchdataxx_C PRIVATE SHAREDEP espeak-ng)
