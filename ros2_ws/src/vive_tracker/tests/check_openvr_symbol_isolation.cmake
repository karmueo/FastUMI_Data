# 本文件检查节点不会在进程启动时全局加载 OpenVR，避免其私有异常符号污染 ROS 依赖。

execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${EXECUTABLE_PATH}"
  RESULT_VARIABLE readelf_result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE readelf_error
)

if(NOT readelf_result EQUAL 0)
  message(FATAL_ERROR "readelf failed: ${readelf_error}")
endif()

if(dynamic_section MATCHES "Shared library: \\[libopenvr_api\\.so\\]")
  message(
    FATAL_ERROR
    "vive_tracker_node must load libopenvr_api.so locally at runtime"
  )
endif()
