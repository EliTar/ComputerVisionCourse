cmake_minimum_required(VERSION 3.2)

project(computervision2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include(./dlib/dlib/cmake)

add_executable(dlibFun dlibFun.cpp)
target_link_libraries(dlibFun ${OpenCV_LIBS} dlib)

add_executable(faceAlignment faceAlignment.cpp)
target_link_libraries(faceAlignment ${OpenCV_LIBS} dlib)