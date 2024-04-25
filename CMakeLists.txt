cmake_minimum_required(VERSION 3.15...3.27)
project(
    AlgebraicMultigrid 
    VERSION 0.1
    DESCRIPTION "C++ algebraic multigrid."
    LANGUAGES CXX)

# FetchContent added in CMake 3.11, downloads during the configure step
# FetchContent_MakeAvailable was added in CMake 3.14; simpler usage
include(FetchContent)

# Let's ensure -std=c++xx instead of -std=g++xx
set(CMAKE_CXX_EXTENSIONS OFF)

# using C++17
set(CMAKE_CXX_STANDARD 17)

# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# allows `cmake --build build --target test` to run all desired tests in test/CMakeLists.txt
# (note: this does not run memcheck)
include(CTest) 

# Allow build documentation
find_package(Doxygen)
if(NOT Doxygen_FOUND)
    message(STATUS "Installing Doxygen via FetchContent")
    FetchContent_Declare(Doxygen GIT_REPOSITORY "git@github.com:doxygen/doxygen.git")
    FetchContent_MakeAvailable(Doxygen)
    message(STATUS "Doxygen installed via FetchContent")
endif()

add_subdirectory(docs)

# Check if eigen3 is available and install if needed
find_package(Eigen3 3.4 NO_MODULE) 
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3")
else()
    message(STATUS "Installing Eigen3 via FetchContent")
    FetchContent_Declare(Eigen3 GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git" GIT_TAG "3.4.0") 
    FetchContent_MakeAvailable(Eigen3)
    message(STATUS "Eigen3 installed via FetchContent")
endif()

# since the library is a template library... then src dir is not feasible
# for virtual funcs?
add_subdirectory(src)
# add_subdirectory(include)

# https://stackoverflow.com/questions/40325957/how-do-i-add-valgrind-tests-to-my-cmake-test-target
add_subdirectory(test)
