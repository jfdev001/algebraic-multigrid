file(
  GLOB 
  AMG_HEADER_LIST 
  CONFIGURE_DEPENDS 
  "${AlgebraicMultigrid_SOURCE_DIR}/include/amg/*.hpp")

# A library is what should be used to link to a file executable app... 
# so probably should make an app dir or something??? though maybe not necessary
# since the python API will call the library here??? in the same way julia
# would use the dll
add_library(amg compat.cpp ${AMG_HEADER_LIST})

# https://stackoverflow.com/questions/11801186/cmake-unable-to-determine-linker-language-with-c
set_target_properties(amg PROPERTIES LINKER_LANGUAGE CXX)

# Allows for `#include <amg/*.hpp>`
target_include_directories(amg PUBLIC ../include)

target_link_libraries(amg PRIVATE Eigen3::Eigen)

target_compile_features(amg PUBLIC)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${AMG_HEADER_LIST})