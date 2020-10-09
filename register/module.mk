LOCAL_PATH := $(call my-dir)


local_lib_src_files :=  register.cpp \
                        ops_kernel_builder_registry.cc \
                        graph_optimizer/graph_fusion/graph_fusion_pass_base.cpp \
                        graph_optimizer/graph_fusion/fusion_pass_registry.cpp \
                        graph_optimizer/graph_fusion/fusion_pattern.cpp \
                        graph_optimizer/graph_fusion/pattern_fusion_base_pass.cpp \
                        graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.cpp \
                        graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h \
                        graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.cpp \
                        graph_optimizer/buffer_fusion/buffer_fusion_pass_base.cpp \
                        graph_optimizer/buffer_fusion/buffer_fusion_pattern.cpp \
                        graph_optimizer/fusion_statistic/fusion_statistic_recorder.cpp \
                        register_format_transfer.cc \
                        op_kernel_registry.cpp \
                        auto_mapping_util.cpp \
                        host_cpu_context.cc \
                        tensor_assign.cpp \
                        scope/scope_graph.cc \
                        scope/scope_pass.cc \
                        scope/scope_pattern.cc \
                        scope/scope_util.cc \
                        scope/scope_pass_registry.cc \
                        ./proto/tensorflow/attr_value.proto \
                        ./proto/tensorflow/function.proto \
                        ./proto/tensorflow/graph.proto \
                        ./proto/tensorflow/node_def.proto \
                        ./proto/tensorflow/op_def.proto \
                        ./proto/tensorflow/resource_handle.proto \
                        ./proto/tensorflow/tensor.proto \
                        ./proto/tensorflow/tensor_shape.proto \
                        ./proto/tensorflow/types.proto \
                        ./proto/tensorflow/versions.proto \
                        ./proto/task.proto \
                        ./proto/om.proto \

local_lib_inc_path :=   \
                        inc \
                        inc/external \
                        inc/external/graph \
                        inc/graph \
                        inc/common \
                        inc/framework \
                        common \
                        common/graph \
                        third_party/protobuf/include \
                        libc_sec/include \
				        third_party/json/include \

tiling_src_files := op_tiling.cpp \

#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libop_tiling_o2

LOCAL_CFLAGS += -std=c++11 -O2
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=

LOCAL_SRC_FILES := $(tiling_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_C_INCLUDES := $(local_lib_inc_path)

include ${BUILD_HOST_STATIC_LIBRARY}


#compiler for host
include $(CLEAR_VARS)
LOCAL_MODULE := libregister

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS :=

LOCAL_WHOLE_STATIC_LIBRARIES := libop_tiling_o2 \

LOCAL_SHARED_LIBRARIES :=   libprotobuf \
                            libc_sec \
                            libslog \
                            libgraph \

LOCAL_SRC_FILES := $(local_lib_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_EXPORT_C_INCLUDE_DIRS := $(generated_sources_dir)/proto/$(LOCAL_PATH)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
LOCAL_C_INCLUDES += LOCAL_EXPORT_C_INCLUDE_DIRS

include ${BUILD_HOST_SHARED_LIBRARY}


#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libop_tiling_o2

LOCAL_CFLAGS += -std=c++11 -O2
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=

LOCAL_SRC_FILES := $(tiling_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
include ${BUILD_STATIC_LIBRARY}


include $(CLEAR_VARS)
LOCAL_MODULE := libregister

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS :=

LOCAL_WHOLE_STATIC_LIBRARIES := libop_tiling_o2

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libprotobuf \
                            libc_sec \
                            libslog \
                            libgraph \

LOCAL_SRC_FILES := $(local_lib_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_EXPORT_C_INCLUDE_DIRS := $(generated_sources_dir)/proto/$(LOCAL_PATH)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
LOCAL_C_INCLUDES += LOCAL_EXPORT_C_INCLUDE_DIRS

include ${BUILD_SHARED_LIBRARY}

#compiler static libregister for host
include $(CLEAR_VARS)
LOCAL_MODULE := libregister

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES := \
    libgraph \
    libprotobuf \

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libslog \

LOCAL_SRC_FILES := $(local_lib_src_files) $(tiling_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_EXPORT_C_INCLUDE_DIRS := $(generated_sources_dir)/proto/$(LOCAL_PATH)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
LOCAL_C_INCLUDES += LOCAL_EXPORT_C_INCLUDE_DIRS

LOCAL_UNINSTALLABLE_MODULE := false
include ${BUILD_HOST_STATIC_LIBRARY}


#compiler static libregister for device
include $(CLEAR_VARS)
LOCAL_MODULE := libregister

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES := \
    libgraph \
    libprotobuf \

LOCAL_SHARED_LIBRARIES := \
    libc_sec \
    libslog \

LOCAL_SRC_FILES := $(local_lib_src_files)  $(tiling_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_EXPORT_C_INCLUDE_DIRS := $(generated_sources_dir)/proto/$(LOCAL_PATH)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
LOCAL_C_INCLUDES += LOCAL_EXPORT_C_INCLUDE_DIRS

LOCAL_UNINSTALLABLE_MODULE := false
include ${BUILD_STATIC_LIBRARY}

#compiler for device
include $(CLEAR_VARS)
LOCAL_MODULE := libregister

LOCAL_CFLAGS += -std=c++11
LOCAL_LDFLAGS :=

LOCAL_STATIC_LIBRARIES :=
LOCAL_SHARED_LIBRARIES :=   libprotobuf \
                            libc_sec \
                            libslog \
                            libgraph \

LOCAL_SRC_FILES := $(local_lib_src_files) $(tiling_src_files)

generated_sources_dir := $(call local-generated-sources-dir)
LOCAL_EXPORT_C_INCLUDE_DIRS := $(generated_sources_dir)/proto/$(LOCAL_PATH)
LOCAL_C_INCLUDES := $(local_lib_inc_path)
LOCAL_C_INCLUDES += LOCAL_EXPORT_C_INCLUDE_DIRS

include ${BUILD_LLT_SHARED_LIBRARY}