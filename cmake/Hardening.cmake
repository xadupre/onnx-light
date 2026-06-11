# Hardening.cmake
#
# Implements the OpenSSF "Compiler Options Hardening Guide for C and C++"
# (https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html)
# as an opt-in set of compile and link flags applied to every onnx_light target.
#
# Enable with ``-DONNX_HARDENING=ON``. The flags below follow the
# "recommended" baseline from the guide for GCC/Clang and MSVC. Each flag is
# probed with check_*_compiler_flag / check_linker_flag and silently skipped
# when the active toolchain does not support it, so the option is safe to turn
# on across a range of compiler versions.
#
# The macro ``onnx_light_apply_hardening(<target>)`` is the public entry point.
# It is a no-op when ONNX_HARDENING is OFF, so call sites can apply it
# unconditionally.

include_guard(GLOBAL)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckLinkerFlag OPTIONAL RESULT_VARIABLE _onnx_have_check_linker_flag)

set(ONNX_LIGHT_HARDENING_COMPILE_OPTIONS "" CACHE INTERNAL "")
set(ONNX_LIGHT_HARDENING_COMPILE_DEFINITIONS "" CACHE INTERNAL "")
set(ONNX_LIGHT_HARDENING_LINK_OPTIONS "" CACHE INTERNAL "")

function(_onnx_light_try_cxx_flag flag out_var)
  string(MAKE_C_IDENTIFIER "ONNX_LIGHT_HARDENING_CXX_${flag}" cache_var)
  check_cxx_compiler_flag("${flag}" ${cache_var})
  if(${cache_var})
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_onnx_light_try_link_flag flag out_var)
  string(MAKE_C_IDENTIFIER "ONNX_LIGHT_HARDENING_LD_${flag}" cache_var)
  if(_onnx_have_check_linker_flag)
    check_linker_flag(CXX "${flag}" ${cache_var})
  else()
    # Fallback for CMake < 3.18: try as a compile+link flag.
    set(CMAKE_REQUIRED_LINK_OPTIONS_BACKUP "${CMAKE_REQUIRED_LINK_OPTIONS}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_cxx_compiler_flag("" ${cache_var})
    set(CMAKE_REQUIRED_LINK_OPTIONS "${CMAKE_REQUIRED_LINK_OPTIONS_BACKUP}")
  endif()
  if(${cache_var})
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

set(_onnx_light_hardening_compile_options "")
set(_onnx_light_hardening_compile_definitions "")
set(_onnx_light_hardening_link_options "")

if(MSVC)
  # OpenSSF guide recommendations for MSVC.
  set(_msvc_compile_candidates
      /GS              # buffer security checks
      /guard:cf        # Control Flow Guard
      /Qspectre        # Spectre v1 mitigations
      /sdl             # additional security checks
      /ZH:SHA_256      # stronger debug hashes
  )
  foreach(flag IN LISTS _msvc_compile_candidates)
    _onnx_light_try_cxx_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _onnx_light_hardening_compile_options "${flag}")
    endif()
  endforeach()

  set(_msvc_link_candidates
      /DYNAMICBASE      # ASLR
      /NXCOMPAT         # DEP
      /CETCOMPAT        # CET shadow stack
      /guard:cf
  )
  foreach(flag IN LISTS _msvc_link_candidates)
    _onnx_light_try_link_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _onnx_light_hardening_link_options "${flag}")
    endif()
  endforeach()
else()
  # GCC / Clang recommendations.
  set(_gcc_compile_candidates
      -Wall
      -Wformat
      -Wformat=2
      -Wimplicit-fallthrough
      -Werror=format-security
      -fstack-protector-strong
      -fstack-clash-protection
      -fcf-protection=full
      -fstrict-flex-arrays=3
      -fno-delete-null-pointer-checks
      -fno-strict-overflow
      -fno-strict-aliasing
      -ftrivial-auto-var-init=zero
      -fexceptions
  )
  foreach(flag IN LISTS _gcc_compile_candidates)
    _onnx_light_try_cxx_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _onnx_light_hardening_compile_options "${flag}")
    endif()
  endforeach()

  # _FORTIFY_SOURCE requires an optimization level (>= -O1); GCC otherwise
  # warns. Use level 3 when available, fall back to 2.
  _onnx_light_try_cxx_flag("-D_FORTIFY_SOURCE=3" _have_fortify3)
  if(_have_fortify3)
    list(APPEND _onnx_light_hardening_compile_definitions "_FORTIFY_SOURCE=3")
  else()
    _onnx_light_try_cxx_flag("-D_FORTIFY_SOURCE=2" _have_fortify2)
    if(_have_fortify2)
      list(APPEND _onnx_light_hardening_compile_definitions "_FORTIFY_SOURCE=2")
    endif()
  endif()
  # libstdc++ runtime assertions (bounds checks on containers).
  list(APPEND _onnx_light_hardening_compile_definitions "_GLIBCXX_ASSERTIONS")

  set(_gcc_link_candidates
      "LINKER:-z,noexecstack"
      "LINKER:-z,relro"
      "LINKER:-z,now"
      "LINKER:-z,nodlopen"
      "LINKER:--as-needed"
      "LINKER:--no-copy-dt-needed-entries"
  )
  foreach(flag IN LISTS _gcc_link_candidates)
    _onnx_light_try_link_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _onnx_light_hardening_link_options "${flag}")
    endif()
  endforeach()
endif()

set(ONNX_LIGHT_HARDENING_COMPILE_OPTIONS "${_onnx_light_hardening_compile_options}"
    CACHE INTERNAL "Hardening compile options resolved for this toolchain.")
set(ONNX_LIGHT_HARDENING_COMPILE_DEFINITIONS "${_onnx_light_hardening_compile_definitions}"
    CACHE INTERNAL "Hardening compile definitions resolved for this toolchain.")
set(ONNX_LIGHT_HARDENING_LINK_OPTIONS "${_onnx_light_hardening_link_options}"
    CACHE INTERNAL "Hardening link options resolved for this toolchain.")

message(STATUS "ONNX_HARDENING: enabled "
        "(compile: ${ONNX_LIGHT_HARDENING_COMPILE_OPTIONS}; "
        "define: ${ONNX_LIGHT_HARDENING_COMPILE_DEFINITIONS}; "
        "link: ${ONNX_LIGHT_HARDENING_LINK_OPTIONS})")

# Apply the resolved hardening flags to a single target. No-op when the lists
# are empty (e.g. compiler did not accept any candidate flag).
function(onnx_light_apply_hardening target)
  if(NOT TARGET ${target})
    return()
  endif()
  if(ONNX_LIGHT_HARDENING_COMPILE_OPTIONS)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:${ONNX_LIGHT_HARDENING_COMPILE_OPTIONS}>)
  endif()
  if(ONNX_LIGHT_HARDENING_COMPILE_DEFINITIONS)
    target_compile_definitions(${target} PRIVATE
        ${ONNX_LIGHT_HARDENING_COMPILE_DEFINITIONS})
  endif()
  if(ONNX_LIGHT_HARDENING_LINK_OPTIONS)
    target_link_options(${target} PRIVATE
        ${ONNX_LIGHT_HARDENING_LINK_OPTIONS})
  endif()
endfunction()
