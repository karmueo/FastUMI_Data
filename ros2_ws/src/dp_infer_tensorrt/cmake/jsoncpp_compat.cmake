# Ubuntu 22.04 的 JsonCpp 使用旧 policy 声明；只在函数局部兼容 CMake 4。
# 导出的 JsonCpp target 属于当前目录，函数结束后仍可由公共 core target 引用。
if(NOT jsoncpp_FOUND)
  function(_dp_infer_tensorrt_find_jsoncpp)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0" AND NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
      set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    endif()
    find_package(jsoncpp REQUIRED)
    set(jsoncpp_FOUND ${jsoncpp_FOUND} PARENT_SCOPE)
  endfunction()
  _dp_infer_tensorrt_find_jsoncpp()
endif()
