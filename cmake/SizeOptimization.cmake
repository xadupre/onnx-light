# SizeOptimization.cmake
#
# Always-on set of compile/link/target flags that reduce the on-disk size of
# the shipped onnx-light shared libraries and Python extension modules while
# keeping debug builds fully debuggable. The behaviour is driven entirely by
# the build configuration (there is no user-facing on/off switch):
#
#   * ``Debug`` / ``RelWithDebInfo`` keep their full symbol table and debug
#     info -- nothing is stripped, so these builds contain everything needed
#     for debugging.
#   * ``Release`` is made as small as possible: the non-dynamic symbol table
#     and debug info are stripped and unnecessary symbols are hidden from the
#     dynamic table.
#
# The dominant contributor to the shipped Linux ``.so`` size is the ELF symbol
# table. The ``lib_onnx_*`` libraries are SHARED and export their symbols so
# the Python extension modules can link against them across library
# boundaries, so ``-fvisibility=hidden`` cannot be applied to them without
# breaking the link. The leaf nanobind modules (``_onnxpy*``, built as
# ``MODULE`` libraries) only need to export their ``PyInit_*`` entry point, so
# they *can* be compiled with fully hidden visibility. The safe, high-impact
# levers are therefore:
#
#   * ``-fvisibility-inlines-hidden`` (VISIBILITY_INLINES_HIDDEN) on every
#     target: inline/template instantiations are emitted in every translation
#     unit that needs them, so hiding them from the dynamic table never breaks
#     cross-library linking but removes a large amount of C++ symbol bloat;
#   * ``-fvisibility=hidden`` (C/CXX_VISIBILITY_PRESET) on the leaf MODULE
#     targets only, so they export nothing but their init symbol;
#   * ``-ffunction-sections -fdata-sections`` + ``--gc-sections`` so the linker
#     drops sections not reachable from any exported/root symbol;
#   * stripping the non-dynamic symbol table and debug info in ``Release`` only
#     (``.dynsym`` is preserved, so dynamic linking is unaffected);
#   * ``/Gy /Gw`` + ``/OPT:REF /OPT:ICF`` for the MSVC toolchain.
#
# Each candidate flag is probed with check_*_compiler_flag / check_linker_flag
# and silently skipped when the active toolchain does not support it, so the
# module is safe across a range of compilers.
#
# The macro ``onnx_light_apply_size_optimization(<target>)`` is the public
# entry point. It inspects the target type to decide whether full hidden
# visibility is safe, and is a no-op for the pieces the toolchain does not
# support, so call sites can apply it unconditionally to every target.

include_guard(GLOBAL)

include(CheckCXXCompilerFlag)
include(CheckLinkerFlag OPTIONAL RESULT_VARIABLE _onnx_size_have_check_linker_flag)

function(_onnx_size_try_cxx_flag flag out_var)
  string(MAKE_C_IDENTIFIER "ONNX_LIGHT_SIZEOPT_CXX_${flag}" cache_var)
  check_cxx_compiler_flag("${flag}" ${cache_var})
  if(${cache_var})
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_onnx_size_try_link_flag flag out_var)
  string(MAKE_C_IDENTIFIER "ONNX_LIGHT_SIZEOPT_LD_${flag}" cache_var)
  if(_onnx_size_have_check_linker_flag)
    check_linker_flag(CXX "${flag}" ${cache_var})
  else()
    # Fallback for CMake < 3.18: try as a compile+link flag.
    set(_backup "${CMAKE_REQUIRED_LINK_OPTIONS}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_cxx_compiler_flag("" ${cache_var})
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_backup}")
  endif()
  if(${cache_var})
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

set(_onnx_size_compile_options "")
set(_onnx_size_link_options "")
set(_onnx_size_link_options_release "")

if(MSVC)
  # Function-level linking (/Gy) and COMDAT for data (/Gw) let the linker
  # discard unreferenced code/data; /OPT:REF removes it and /OPT:ICF folds
  # identical COMDATs. /OPT:* conflict with incremental linking, so they are
  # confined to the Release configuration.
  foreach(flag IN ITEMS /Gy /Gw)
    _onnx_size_try_cxx_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _onnx_size_compile_options "${flag}")
    endif()
  endforeach()

  foreach(flag IN ITEMS /OPT:REF /OPT:ICF)
    _onnx_size_try_link_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _onnx_size_link_options_release "${flag}")
    endif()
  endforeach()
else()
  # GCC / Clang: emit one section per function/data object so the linker can
  # garbage-collect the ones no exported symbol keeps alive.
  foreach(flag IN ITEMS -ffunction-sections -fdata-sections)
    _onnx_size_try_cxx_flag("${flag}" _ok)
    if(_ok)
      list(APPEND _onnx_size_compile_options "${flag}")
    endif()
  endforeach()

  # Section garbage collection: GNU ld / lld use --gc-sections, Apple ld64
  # uses -dead_strip.
  _onnx_size_try_link_flag("-Wl,--gc-sections" _have_gc)
  if(_have_gc)
    list(APPEND _onnx_size_link_options "-Wl,--gc-sections")
  else()
    _onnx_size_try_link_flag("-Wl,-dead_strip" _have_dead_strip)
    if(_have_dead_strip)
      list(APPEND _onnx_size_link_options "-Wl,-dead_strip")
    endif()
  endif()

  # Strip the symbol table and debug info in Release only. Prefer the full
  # strip (-s / --strip-all); fall back to stripping local symbols (-Wl,-x)
  # on toolchains that reject it (e.g. Apple ld64).
  _onnx_size_try_link_flag("-Wl,--strip-all" _have_strip_all)
  if(_have_strip_all)
    list(APPEND _onnx_size_link_options_release "-Wl,--strip-all")
  else()
    _onnx_size_try_link_flag("-Wl,-x" _have_strip_local)
    if(_have_strip_local)
      list(APPEND _onnx_size_link_options_release "-Wl,-x")
    endif()
  endif()
endif()

set(ONNX_LIGHT_SIZEOPT_COMPILE_OPTIONS "${_onnx_size_compile_options}"
    CACHE INTERNAL "Size-optimization compile options resolved for this toolchain.")
set(ONNX_LIGHT_SIZEOPT_LINK_OPTIONS "${_onnx_size_link_options}"
    CACHE INTERNAL "Size-optimization link options resolved for this toolchain.")
set(ONNX_LIGHT_SIZEOPT_LINK_OPTIONS_RELEASE "${_onnx_size_link_options_release}"
    CACHE INTERNAL "Size-optimization link options applied only to the Release configuration.")

message(STATUS "onnx-light size optimization "
        "(compile: ${ONNX_LIGHT_SIZEOPT_COMPILE_OPTIONS}; "
        "link: ${ONNX_LIGHT_SIZEOPT_LINK_OPTIONS}; "
        "link[Release]: ${ONNX_LIGHT_SIZEOPT_LINK_OPTIONS_RELEASE})")

# Apply the resolved size-optimization flags to a single target.
#
# Visibility is applied to every target (safe in all configurations, as it
# affects the *dynamic* symbol table, never the DWARF/PDB debug info):
#   * VISIBILITY_INLINES_HIDDEN is safe everywhere.
#   * Full hidden visibility (C/CXX_VISIBILITY_PRESET hidden) is only set on
#     leaf MODULE targets (the nanobind extensions) and executables, which
#     export nothing but their entry point. Setting it on the SHARED
#     lib_onnx_* libraries would hide the symbols the extension modules link
#     against and break the build, so those keep default visibility.
#
# Stripping is confined to the Release configuration via a generator
# expression, so Debug / RelWithDebInfo keep their full symbol table and debug
# info. The function is a no-op for pieces the toolchain does not support, so
# it is safe to call unconditionally on every target.
function(onnx_light_apply_size_optimization target)
  if(NOT TARGET ${target})
    return()
  endif()

  set_target_properties(${target} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

  get_target_property(_onnx_size_type ${target} TYPE)
  if(_onnx_size_type STREQUAL "MODULE_LIBRARY" OR _onnx_size_type STREQUAL "EXECUTABLE")
    set_target_properties(${target} PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden)
  endif()

  if(ONNX_LIGHT_SIZEOPT_COMPILE_OPTIONS)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:${ONNX_LIGHT_SIZEOPT_COMPILE_OPTIONS}>)
  endif()
  if(ONNX_LIGHT_SIZEOPT_LINK_OPTIONS)
    target_link_options(${target} PRIVATE ${ONNX_LIGHT_SIZEOPT_LINK_OPTIONS})
  endif()
  if(ONNX_LIGHT_SIZEOPT_LINK_OPTIONS_RELEASE)
    target_link_options(${target} PRIVATE
        $<$<CONFIG:Release>:${ONNX_LIGHT_SIZEOPT_LINK_OPTIONS_RELEASE}>)
  endif()
endfunction()
