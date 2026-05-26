# Modified copy of
#   https://github.com/sdpython/onnx-extended/blob/main/_cmake/externals/FindOrt.cmake
# Trimmed to the bits this example needs: download a prebuilt onnxruntime
# release archive (CPU only) and expose ``ONNXRUNTIME_INCLUDE_DIR`` plus
# ``ONNXRUNTIME_LIBRARY``. CUDA / custom-op helpers from the original module
# are intentionally omitted.

include(FetchContent)

if(NOT ORT_VERSION)
  set(ORT_VERSION 1.19.2)
endif()

if(MSVC)
  set(_ORT_ARCHIVE "onnxruntime-win-x64-${ORT_VERSION}.zip")
  set(_ORT_FOLDER  "onnxruntime-win-x64-${ORT_VERSION}")
elseif(APPLE)
  set(_ORT_ARCHIVE "onnxruntime-osx-universal2-${ORT_VERSION}.tgz")
  set(_ORT_FOLDER  "onnxruntime-osx-universal2-${ORT_VERSION}")
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ORT_ARCHIVE "onnxruntime-linux-aarch64-${ORT_VERSION}.tgz")
    set(_ORT_FOLDER  "onnxruntime-linux-aarch64-${ORT_VERSION}")
  else()
    set(_ORT_ARCHIVE "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
    set(_ORT_FOLDER  "onnxruntime-linux-x64-${ORT_VERSION}")
  endif()
endif()

set(_ORT_URL
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ORT_ARCHIVE}")

message(STATUS "ORT - retrieving release ${ORT_VERSION} from ${_ORT_URL}")
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(onnxruntime URL "${_ORT_URL}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_SOURCE_DIR}/include")
set(ONNXRUNTIME_LIB_DIR     "${onnxruntime_SOURCE_DIR}/lib")
set(ONNXRUNTIME_ROOT_DIR    "${onnxruntime_SOURCE_DIR}" CACHE PATH
    "Root of the downloaded onnxruntime release" FORCE)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS "${ONNXRUNTIME_LIB_DIR}"
  NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_LIBRARY OR NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "FindOrt: failed to locate downloaded onnxruntime at '${onnxruntime_SOURCE_DIR}'. "
    "Expected '${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h' and "
    "libonnxruntime under '${ONNXRUNTIME_LIB_DIR}'.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ort
  REQUIRED_VARS ONNXRUNTIME_LIBRARY ONNXRUNTIME_INCLUDE_DIR
  VERSION_VAR   ORT_VERSION)
