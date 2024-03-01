cmake_minimum_required(VERSION 3.15...3.27)
project(
    AlgebraicMultigrid 
    VERSION 0.1
    DESCRIPTION "C++ algebraic multigrid backend for use with Python interface"
    LANGUAGES CXX)

# TODO: How does this work with pybindings???
# Only do these if this is the main project, and not if it is included through add_subdirectory
# if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

# Let's ensure -std=c++xx instead of -std=g++xx
set(CMAKE_CXX_EXTENSIONS OFF)

# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(CTest)

# Docs only available if this is the main app
find_package(Doxygen)
if(Doxygen_FOUND)
    add_subdirectory(docs)
else()
    message(STATUS "Doxygen not found, not building docs")
endif()

# FetchContent added in CMake 3.11, downloads during the configure step
# FetchContent_MakeAvailable was added in CMake 3.14; simpler usage
include(FetchContent)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(src)

add_subdirectory(test)