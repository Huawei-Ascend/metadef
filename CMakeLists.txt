cmake_minimum_required(VERSION 3.14)
project (MetaDef[CXX])

set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR})

option(ENABLE_OPEN_SRC "Enable graphengine compile in opensource." FALSE)

if (ENABLE_OPEN_SRC)
    set(HI_PYTHON python3.7)

    include(cmake/external_libs/protobuf_shared.cmake)
    include(cmake/external_libs/protoc.cmake)
    include(cmake/external_libs/securec.cmake)
    include(cmake/external_libs/json.cmake)
    include(cmake/FindModule.cmake)
    include(cmake/intf_pub_linux.cmake)

    if(DEFINED ENV{ASCEND_CUSTOM_PATH})
        set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_DIR /usr/local/Ascend)
    endif()

    set(ASCEND_DRIVER_DIR ${ASCEND_DIR}/driver/lib64/common)
    set(ASCEND_RUNTIME_DIR ${ASCEND_DIR}/fwkacllib/lib64)

    find_module(slog libslog.so ${ASCEND_DRIVER_DIR})
    find_module(mmpa libmmpa.so ${ASCEND_DRIVER_DIR})
    find_module(error_manager liberror_manager.so ${ASCEND_RUNTIME_DIR})

endif()

add_subdirectory(graph)
add_subdirectory(register)
