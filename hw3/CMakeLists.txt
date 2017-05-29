cmake_minimum_required(VERSION 3.2)
project(hw3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hw3 hw3.cpp)
target_link_libraries(hw3 ${OpenCV_LIBS})