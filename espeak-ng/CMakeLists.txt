cmake_minimum_required(VERSION 3.8)

project(espeak-ng)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/deps.cmake)
include(cmake/config.cmake)
add_subdirectory(src)
include(cmake/data.cmake)

target_include_directories(espeak-ng INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/include/espeak-ng)
