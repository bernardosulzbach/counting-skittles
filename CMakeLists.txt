cmake_minimum_required(VERSION 3.11)

project(counting-skittles)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_definitions(-Wall)
add_definitions(-Wextra)
add_definitions(-Wshadow)
add_definitions(-Werror)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DDEBUGGING)
endif()

add_executable(count src/Count.cpp)

find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework)
include_directories(${Boost_INCLUDE_DIR})
set(Boost_USE_MULTITHREADED ON)

find_package(PkgConfig REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(count ${OpenCV_LIBS})
target_link_libraries(count ${Boost_LIBRARIES})

if(IPO_SUPPORTED)
  set_property(TARGET solver PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
