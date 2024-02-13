add_library(ucd STATIC
  src/case.c
  src/categories.c
  src/ctype.c
  src/proplist.c
  src/scripts.c
  src/tostring.c
)
target_include_directories(ucd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
