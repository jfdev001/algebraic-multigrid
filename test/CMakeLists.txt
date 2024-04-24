# Testing library
FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.6)
FetchContent_MakeAvailable(catch)
# Adds Catch2::Catch2

# Tests need to be added as executables first
add_executable(testlib testlib.cpp)

# possibly needed for macos and windows? 
# https://stackoverflow.com/questions/19829281/cmake-os-x-failure-ar-no-archive-members-specific
target_include_directories(testlib PRIVATE "${AlgebraicMultigrid_SOURCE_DIR}/include")

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(testlib PRIVATE amg Catch2::Catch2 Eigen3::Eigen)

# If you register a test, then ctest and make test will run it.
# You can also run examples and check the output, as well.
add_test(NAME testlibtest COMMAND testlib) # Command can be a target
