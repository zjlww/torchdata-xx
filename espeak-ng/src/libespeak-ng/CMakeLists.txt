set(ESPEAK_CONFIG_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(ESPEAK_CONFIG_H ${ESPEAK_CONFIG_DIR}/config.h)
configure_file(config.h.in ${ESPEAK_CONFIG_H})

add_library(espeak-ng-config INTERFACE)
target_include_directories(espeak-ng-config INTERFACE ${ESPEAK_CONFIG_DIR})

add_library(espeak-ng STATIC
  common.cpp
  mnemonics.cpp
  error.cpp
  ieee80.cpp

  context.cpp

  compiledata.cpp
  compiledict.cpp

  dictionary.cpp
  encoding.cpp
  intonation.cpp
  langopts.cpp
  numbers.cpp
  phoneme.cpp
  phonemelist.cpp
  readclause.cpp
  setlengths.cpp
  soundicon.cpp
  spect.cpp
  ssml.cpp
  synthdata.cpp
  synthesize.cpp
  tr_languages.cpp
  translate.cpp
  translateword.cpp
  voices.cpp
  wavegen.cpp
  speech.cpp

  espeak_api.cpp
)
set_target_properties(
  espeak-ng PROPERTIES
  CXX_STANDARD 17
  MACOSX_RPATH ON
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

target_include_directories(espeak-ng BEFORE PRIVATE $<TARGET_PROPERTY:espeak-include,INTERFACE_INCLUDE_DIRECTORIES>)

if (NOT MSVC)
  target_compile_options(espeak-ng PRIVATE
    "-fPIC"
    "-fvisibility=hidden"
    "-fno-exceptions"
    "-fwrapv"

    "-pedantic"

    "-Wunused-parameter"
    "-Wunused"
    "-Wuninitialized"
    "-Wreturn-type"
    # "-Wmissing-prototypes"
    # "-Wint-conversion"
    # "-Wimplicit"
    "-Wmisleading-indentation"
  )
endif()

target_compile_definitions(espeak-ng PRIVATE "LIBESPEAK_NG_EXPORT=1")
if (NOT BUILD_SHARED_LIBS)
  target_compile_definitions(espeak-ng INTERFACE "LIBESPEAK_NG_EXPORT=1")
endif()

target_compile_definitions(espeak-ng PRIVATE "PATH_ESPEAK_DATA=\"${CMAKE_INSTALL_PREFIX}/share/espeak-ng-data\"")

target_link_libraries(espeak-ng PRIVATE espeak-ng-config ucd)

if (NOT MSVC)
  target_link_libraries(espeak-ng PRIVATE m)
endif()
target_link_libraries(espeak-ng PRIVATE espeak-include)


if ((MINGW OR WIN32 OR MSVC) AND BUILD_SHARED_LIBS)
  set_target_properties(espeak-ng PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ..
  )
endif()

if (MINGW)
  target_link_options(espeak-ng PRIVATE "-static-libstdc++" "-static")
endif()

set_target_properties(espeak-ng PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
  VERSION ${PROJECT_VERSION}
)

install(TARGETS espeak-ng LIBRARY)
