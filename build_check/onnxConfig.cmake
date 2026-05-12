
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was onnxConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
get_filename_component(_onnx_cmake_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# OpenMP remains optional for the compatibility package, matching onnx_light.
find_dependency(onnx_light CONFIG REQUIRED
  PATHS "${_onnx_cmake_root}/onnx_light"
  NO_DEFAULT_PATH
)

if(NOT TARGET onnx::onnx)
  add_library(onnx::onnx INTERFACE IMPORTED)
  set_target_properties(onnx::onnx PROPERTIES
    INTERFACE_LINK_LIBRARIES onnx_light::onnx_light
  )
endif()

check_required_components(onnx)
