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

# Other Dependencies
find_package(pybind11 REQUIRED)

# Python:
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# PyTorch:
include(cmake/torch.cmake)

file(GLOB SRC_CPP "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp")

# Python Extension:
pybind11_add_module(torchdataxx binding.cpp ${SRC_CPP})
target_link_libraries(torchdataxx PRIVATE Torch)
target_include_directories(torchdataxx PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/csrc/")

# Testing Target:
add_executable(test test.cpp ${SRC_CPP})
target_link_libraries(test PRIVATE Torch)
target_include_directories(test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/csrc/")