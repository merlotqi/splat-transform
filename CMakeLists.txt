cmake_minimum_required(VERSION 3.10)
project(splat-transform)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# find_package(Eigen3 CONFIG REQUIRED)

add_library(splat_trans STATIC)

add_subdirectory(utils)
add_subdirectory(readers)

# target_link_libraries(splat_trans Eigen3::Eigen)