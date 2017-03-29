cmake_minimum_required(VERSION 3.2)

project(computervision1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hw1 hw1.cpp slic.cpp)
target_link_libraries(hw1 ${OpenCV_LIBS})

if (CMAKE_COMPILER_IS_GNUCC)
	set_property( TARGET hw1 APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )
endif (CMAKE_COMPILER_IS_GNUCC)
