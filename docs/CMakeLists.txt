set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
# https://stackoverflow.com/questions/12147668/doxygen-private-function
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)

doxygen_add_docs(
    docs 
    include/amg
    "${CMAKE_CURRENT_SOURCE_DIR}/mainpage.md"
    WORKING_DIRECTORY 
    "${PROJECT_SOURCE_DIR}")
