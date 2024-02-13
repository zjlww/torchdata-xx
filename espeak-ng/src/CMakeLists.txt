add_library(espeak-include INTERFACE)
target_include_directories(espeak-include INTERFACE include include/compat)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ucd-tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libespeak-ng)

add_executable(espeak-ng-bin espeak-ng.cpp)
set_target_properties(espeak-ng-bin PROPERTIES OUTPUT_NAME espeak-ng)
target_link_libraries(
  espeak-ng-bin PRIVATE espeak-ng espeak-ng-config ucd
)
set_target_properties(
  espeak-ng-bin PROPERTIES
  MACOSX_RPATH ON
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)
if (MINGW)
  target_link_options(espeak-ng-bin PRIVATE "-static-libstdc++" "-static")
endif()
if (MSVC)
  target_sources(espeak-ng-bin PRIVATE compat/getopt.c)
endif()
if (NOT WIN32)
  add_custom_target(
    speak-ng-link ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink espeak-ng ${CMAKE_CURRENT_BINARY_DIR}/speak-ng
    COMMAND ${CMAKE_COMMAND} -E create_symlink espeak-ng ${CMAKE_CURRENT_BINARY_DIR}/espeak
    COMMAND ${CMAKE_COMMAND} -E create_symlink espeak-ng ${CMAKE_CURRENT_BINARY_DIR}/speak
    COMMENT "Link espeak-ng to compat names"
    DEPENDS espeak-ng-bin
  )
endif()