cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(sac)

# Set C++ standard and architecture (for macOS M-series compatibility)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)

# Enable debug mode and generate compile_commands.json
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O2")

# LibTorch and Eigen paths
set(CMAKE_PREFIX_PATH "/Users/abc/libtorch/libtorch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories("/opt/homebrew/include/eigen3")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    TrainSAC.cpp
    TestPPO.cpp
    ReplayBuffer.cpp
)

# Executable: SAC training
add_executable(train_sac TrainSAC.cpp ReplayBuffer.cpp)
target_link_libraries(train_sac ${TORCH_LIBRARIES})
target_include_directories(train_sac PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# Dummy target for header indexing in IDEs
add_custom_target(force_headers_indexing SOURCES
    Models.h
    SoftActorCritic.h
    TestEnvironment.h
    ReplayBuffer.h
    MultiAgentEnvironment.h
    MultiSoftActorCritic.h
)
