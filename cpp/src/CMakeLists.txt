file(
    GLOB 
    HEADER_LIST 
    CONFIGURE_DEPENDS "${AlgebraicMultigrid_SOURCE_DIR}/include/modern/*.hpp")

# TODO: should there be multiple cpps here?? or is the library
# a single file?? how to handle
add_library(amg_library multigrid.cpp ${HEADER_LIST})

target_include_directories(amg_library PUBLIC ../include)

target_link_libraries(amg_library PRIVATE Eigen3::Eigen)

target_compile_features(modern_library PUBLIC cxx_std_11)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
