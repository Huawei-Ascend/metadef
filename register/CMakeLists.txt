set(PROTO_LIST
    "${TOP_DIR}/inc/register/proto/tensorflow/attr_value.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/function.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/graph.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/node_def.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/op_def.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/resource_handle.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/tensor.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/tensor_shape.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/types.proto"
    "${TOP_DIR}/inc/register/proto/tensorflow/versions.proto"
    "${TOP_DIR}/inc/common/proto/task.proto"
    "${TOP_DIR}/inc/common/proto/om.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

set(SRC_LIST
    "register.cpp"
    "graph_optimizer/graph_fusion/graph_fusion_pass_base.cpp"
    "graph_optimizer/graph_fusion/fusion_pass_registry.cpp"
    "graph_optimizer/graph_fusion/fusion_pattern.cpp"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass.cpp"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.cpp"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
    "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.cpp"
    "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.cpp"
    "graph_optimizer/buffer_fusion/buffer_fusion_pattern.cpp"
    "graph_optimizer/fusion_statistic/fusion_statistic_recorder.cpp"
    "register_format_transfer.cc"
    "op_kernel_registry.cpp"
    "auto_mapping_util.cpp"
    "host_cpu_context.cc"
    "tensor_assign.cpp"
    "scope/scope_graph.cc"
    "scope/scope_pass.cc"
    "scope/scope_pattern.cc"
    "scope/scope_util.cc"
    "scope/scope_pass_registry.cc"
)

############ libregister.so ############
add_library(register SHARED ${SRC_LIST} ${PROTO_SRCS})

target_include_directories(register PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${TOP_DIR}
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/inc/external/graph
    ${TOP_DIR}/inc/graph
    ${TOP_DIR}/inc/common
    ${TOP_DIR}/inc/framework
    ${TOP_DIR}/common
    ${TOP_DIR}/common/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
)

target_link_libraries(register PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--whole-archive
    op_tiling_o2
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    protobuf
    c_sec
    slog
    graph
    -Wl,--as-needed
    json
)

############ libregister.a ############
add_library(register_static STATIC ${SRC_LIST} ${PROTO_SRCS})

target_include_directories(register_static PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${TOP_DIR}
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/inc/external/graph
    ${TOP_DIR}/inc/graph
    ${TOP_DIR}/inc/common
    ${TOP_DIR}/inc/framework
    ${TOP_DIR}/common
    ${TOP_DIR}/common/graph
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/ge
)

target_link_libraries(register_static PRIVATE
    protobuf
    c_sec
    json
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(register_static PROPERTIES
    OUTPUT_NAME register
)


############ libop_tiling_o2.a ############
add_library(op_tiling_o2 STATIC
    "op_tiling.cpp"
)

target_include_directories(op_tiling_o2 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${TOP_DIR}
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/inc/external/graph
    ${TOP_DIR}/inc/graph
    ${TOP_DIR}/inc/common
    ${TOP_DIR}/inc/framework
    ${TOP_DIR}/common
    ${TOP_DIR}/common/graph
)

target_compile_options(op_tiling_o2 PRIVATE
    -O2
)

target_link_libraries(op_tiling_o2 PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    json
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS register register_static OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
)
