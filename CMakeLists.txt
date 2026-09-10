cmake_minimum_required(VERSION 3.16)
project(ML CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(MLlib INTERFACE)
target_include_directories(MLlib INTERFACE include)

enable_testing()
add_subdirectory(tests)