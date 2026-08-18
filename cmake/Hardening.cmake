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

function(_onnx_light_normalize_msvc_arch arch out_var)
  string(TOLOWER "${arch}" _arch)
  if(_arch STREQUAL "win32" OR _arch STREQUAL "x86")
    set(${out_var} "x86" PARENT_SCOPE)
  elseif(_arch STREQUAL "x64" OR _arch STREQUAL "amd64" OR _arch STREQUAL "x86_64")
    set(${out_var} "x64" PARENT_SCOPE)
  elseif(_arch STREQUAL "arm")
    set(${out_var} "arm" PARENT_SCOPE)
  elseif(_arch STREQUAL "arm64" OR _arch STREQUAL "aarch64")
    set(${out_var} "arm64" PARENT_SCOPE)
  elseif(_arch STREQUAL "arm64ec")
    set(${out_var} "arm64ec" PARENT_SCOPE)
  else()
    set(${out_var} "${_arch}" PARENT_SCOPE)
  endif()
endfunction()

function(_onnx_light_get_msvc_spectre_lib_dir out_var)
  if(NOT DEFINED ENV{VCToolsInstallDir} OR "$ENV{VCToolsInstallDir}" STREQUAL "")
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  if(CMAKE_VS_PLATFORM_NAME)
    set(_target_arch "${CMAKE_VS_PLATFORM_NAME}")
  elseif(CMAKE_GENERATOR_PLATFORM)
    set(_target_arch "${CMAKE_GENERATOR_PLATFORM}")
  elseif(DEFINED ENV{VSCMD_ARG_TGT_ARCH} AND NOT "$ENV{VSCMD_ARG_TGT_ARCH}" STREQUAL "")
    set(_target_arch "$ENV{VSCMD_ARG_TGT_ARCH}")
  elseif(CMAKE_SYSTEM_PROCESSOR)
    set(_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
  else()
    set(_target_arch "x64")
  endif()
  _onnx_light_normalize_msvc_arch("${_target_arch}" _target_arch)

  file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" _vctools_dir)
  set(_spectre_dir "${_vctools_dir}/lib/spectre/${_target_arch}")
  if(IS_DIRECTORY "${_spectre_dir}")
    set(${out_var} "${_spectre_dir}" PARENT_SCOPE)
    return()
  endif()

  if(_target_arch STREQUAL "arm64ec")
    set(_arm64_fallback "${_vctools_dir}/lib/spectre/arm64")
    if(IS_DIRECTORY "${_arm64_fallback}")
      set(${out_var} "${_arm64_fallback}" PARENT_SCOPE)
      return()
    endif()
  endif()

  set(${out_var} "" PARENT_SCOPE)
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

  list(FIND _onnx_light_hardening_compile_options "/Qspectre" _onnx_light_qspectre_index)
  if(NOT _onnx_light_qspectre_index EQUAL -1)
    _onnx_light_get_msvc_spectre_lib_dir(_onnx_light_msvc_spectre_lib_dir)
    if(_onnx_light_msvc_spectre_lib_dir)
      list(APPEND _onnx_light_hardening_link_options
           "SHELL:/LIBPATH:\"${_onnx_light_msvc_spectre_lib_dir}\"")
    else()
      message(WARNING
              "ONNX_HARDENING: /Qspectre is enabled but the MSVC Spectre-mitigated "
              "CRT/STL libraries could not be located. The resulting binaries may "
              "fail Spectre-mitigation validation. Install the C++ "
              "Spectre-mitigated libs Visual Studio component and build from a "
              "Developer Command Prompt.")
    endif()
  endif()
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
  #
  # Some toolchains (notably the Red Hat gcc-toolset shipped in the
  # manylinux_2_28 images used for the release wheels) already predefine
  # ``_FORTIFY_SOURCE`` in their default spec. Passing ``-D_FORTIFY_SOURCE=...``
  # then emits a ``"_FORTIFY_SOURCE" redefined`` warning for every translation
  # unit. Emit the level as an ``-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=<n>`` pair
  # of compile options (rather than a bare compile definition) so the undefine
  # immediately precedes the redefinition on the command line, silencing the
  # redefinition warning while still enabling the hardened level.
  _onnx_light_try_cxx_flag("-D_FORTIFY_SOURCE=3" _have_fortify3)
  if(_have_fortify3)
    list(APPEND _onnx_light_hardening_compile_options
         "-U_FORTIFY_SOURCE" "-D_FORTIFY_SOURCE=3")
  else()
    _onnx_light_try_cxx_flag("-D_FORTIFY_SOURCE=2" _have_fortify2)
    if(_have_fortify2)
      list(APPEND _onnx_light_hardening_compile_options
           "-U_FORTIFY_SOURCE" "-D_FORTIFY_SOURCE=2")
    endif()
  endif()
  # libstdc++ runtime assertions (bounds checks on containers).
  list(APPEND _onnx_light_hardening_compile_definitions "_GLIBCXX_ASSERTIONS")

  set(_gcc_link_candidates
      "LINKER:-z,noexecstack"
      "LINKER:-z,relro"
      "LINKER:-z,now"
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
