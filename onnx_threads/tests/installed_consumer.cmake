function(run_checked)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer command failed (${result}): ${ARGV}")
  endif()
endfunction()

set(prefix "${BUILD_DIR}/consumer-prefix")
set(consumer_build "${BUILD_DIR}/installed-consumer")
run_checked("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}" --config "${CONFIG}")
run_checked("${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${consumer_build}"
    -G "${GENERATOR}" "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_PREFIX_PATH=${prefix}")
run_checked("${CMAKE_COMMAND}" --build "${consumer_build}" --config "${CONFIG}" --parallel 6)
run_checked("${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build}"
    -C "${CONFIG}" --output-on-failure)
