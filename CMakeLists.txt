cmake_minimum_required(VERSION 3.9)
project(name_entities_openMP)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

include(/home/giulia/Scaricati/MITIE-master/mitielib/cmake)

add_executable(name_entities_openMP main.cpp Histogram.cpp Histogram.h StringAdjuster.cpp StringAdjuster.h)

TARGET_LINK_LIBRARIES(name_entities_openMP mitie)
