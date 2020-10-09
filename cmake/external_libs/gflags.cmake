if (HAVE_GFLAGS)
    return()
endif()

include(ExternalProject)
#set(CMAKE_INSTALL_PREFIX ${METADEF_DIR}/output)

if ((${CMAKE_INSTALL_PREFIX} STREQUAL /usr/local) OR
    (${CMAKE_INSTALL_PREFIX} STREQUAL "C:/Program Files (x86)/ascend"))
    set(CMAKE_INSTALL_PREFIX ${METADEF_DIR}/output CACHE STRING "path for install()" FORCE)
    message(STATUS "No install prefix selected, default to ${CMAKE_INSTALL_PREFIX}.")
endif()

ExternalProject_Add(gflags_build
                    #URL http://tfk.inhuawei.com/api/containers/container1/download/protobuf-3.8.0.tar.gz
                    #URL /home/txd/workspace/linux_cmake/pkg/protobuf-3.8.0.tar.gz
                    SOURCE_DIR ${METADEF_DIR}/../third_party/gflags/src/gflags-2.2.2 
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/gflags <SOURCE_DIR>
                    BUILD_COMMAND $(MAKE)
                    INSTALL_COMMAND $(MAKE) install
                    EXCLUDE_FROM_ALL TRUE 
)

set(GFLAGS_PKG_DIR ${CMAKE_INSTALL_PREFIX}/gflags)

add_library(gflags_static STATIC IMPORTED)

set_target_properties(gflags_static PROPERTIES
                      IMPORTED_LOCATION ${GFLAGS_PKG_DIR}/lib/libgflags.a
)

add_library(gflags INTERFACE)
target_include_directories(gflags INTERFACE ${GFLAGS_PKG_DIR}/include)
target_link_libraries(gflags INTERFACE gflags_static)

add_dependencies(gflags gflags_build)

set(HAVE_GFLAGS TRUE CACHE BOOL "gflags build add")