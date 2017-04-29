cmake_minimum_required(VERSION 3.2)

project(computervision2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hw2 hw2.cpp)
target_link_libraries(hw2 ${OpenCV_LIBS})

add_executable(andrey andrey.cpp)
target_link_libraries(andrey ${OpenCV_LIBS})

if (CMAKE_COMPILER_IS_GNUCC)
	set_property( TARGET hw2 APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )
endif (CMAKE_COMPILER_IS_GNUCC)