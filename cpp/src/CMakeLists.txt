# Store all hpp files in HEADER_LIST
file(
  GLOB 
  HEADER_LIST 
  CONFIGURE_DEPENDS "${AlgebraicMultigrid_SOURCE_DIR}/include/amg/*.hpp")

# is this right?
# store all cpp files in SRC LIST
file(
  GLOB 
  SRC_LIST
  CONFIGURE_DEPENDS "${AlgebraicMultigrid_SOURCE_DIR}/src/*.cpp")

# TODO: should there be multiple cpps here?? or is the library
# a single file?? how to handle
# a library is what should be used to link to a file executable app... 
# so probably should make an app dir or something??? though maybe not necessary
# since the python API will call the library here??? in the same way julia
# would use the dll
add_library(amg ${SRC_LIST} ${HEADER_LIST})

target_include_directories(amg PUBLIC ../include)

target_link_libraries(amg PRIVATE Eigen3::Eigen)

target_compile_features(amg PUBLIC cxx_std_11)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
