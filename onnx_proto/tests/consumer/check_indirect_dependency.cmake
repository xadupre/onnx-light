function(read_dynamic path output)
  execute_process(COMMAND "${READELF}" -d "${path}"
      RESULT_VARIABLE result OUTPUT_VARIABLE dynamic ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot read dynamic dependencies of ${path}: ${error}")
  endif()
  set(${output} "${dynamic}" PARENT_SCOPE)
endfunction()

read_dynamic("${CONSUMER}" consumer_dynamic)
read_dynamic("${PROTO}" proto_dynamic)
string(FIND "${consumer_dynamic}" "Shared library: [${PROTO_NAME}]" proto_needed)
string(FIND "${consumer_dynamic}" "Shared library: [${THREADS_NAME}]" threads_needed)
string(FIND "${proto_dynamic}" "Shared library: [${THREADS_NAME}]" indirect_needed)
if(proto_needed EQUAL -1 OR NOT threads_needed EQUAL -1 OR indirect_needed EQUAL -1)
  message(FATAL_ERROR
      "Expected consumer -> proto -> threads, without a direct consumer -> threads dependency.\n"
      "Consumer:\n${consumer_dynamic}\nProto:\n${proto_dynamic}")
endif()
if(NOT consumer_dynamic MATCHES "\\(RUNPATH\\)" OR consumer_dynamic MATCHES "\\(RPATH\\)")
  message(FATAL_ERROR "The consumer must use RUNPATH, not transitive RPATH.")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    --unset=LD_LIBRARY_PATH --unset=LD_PRELOAD "${CONSUMER}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Installed out-of-line proto consumer failed (${result}).")
endif()
