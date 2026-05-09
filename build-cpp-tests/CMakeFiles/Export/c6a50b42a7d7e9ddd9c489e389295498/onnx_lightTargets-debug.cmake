#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "onnx_light::onnx_light" for configuration "Debug"
set_property(TARGET onnx_light::onnx_light APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(onnx_light::onnx_light PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/liblib_onnx_cpp.a"
  )

list(APPEND _cmake_import_check_targets onnx_light::onnx_light )
list(APPEND _cmake_import_check_files_for_onnx_light::onnx_light "${_IMPORT_PREFIX}/lib/liblib_onnx_cpp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
