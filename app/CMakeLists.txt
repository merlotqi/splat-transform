set(GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags CONFIG REQUIRED)

add_executable(SplatTransform main.cpp)

target_link_libraries(SplatTransform PRIVATE gflags::gflags splat)
