cmake_minimum_required(VERSION 3.16.3)
project(kanachan)

enable_testing()

message("PYTHON_VERSION=${PYTHON_VERSION}")
message("PYTHON_INCLUDE_PATH=${PYTHON_INCLUDE_PATH}")

message("CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
message("CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_STANDARD 20)

include_directories(
  "${PYTHON_INCLUDE_PATH}"
  src)

if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
  add_compile_options(-Werror)
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(KANACHAN_ENABLE_ASSERT ON CACHE BOOL "Enable assertions.")
endif()
if (KANACHAN_ENABLE_ASSERT)
  add_compile_definitions(KANACHAN_ENABLE_ASSERT)
endif()
message("KANACHAN_ENABLE_ASSERT=${KANACHAN_ENABLE_ASSERT}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    add_compile_definitions(_GLIBCXX_DEBUG _GLIBCXX_DEBUG_PEDANTIC)
  endif()
endif()

if (KANACHAN_WITH_COVERAGE)
  add_compile_definitions(KANACHAN_WITH_COVERAGE)
  add_compile_options("-coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -coverage")
endif()
message("KANACHAN_WITH_COVERAGE=${KANACHAN_WITH_COVERAGE}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    set(KANACHAN_WITH_ASAN ON CACHE BOOL "With AddressSanitizer.")
  endif()
endif()
if (KANACHAN_WITH_ASAN)
  add_compile_options(-fsanitize=address)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()
message("KANACHAN_WITH_ASAN=${KANACHAN_WITH_ASAN}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    set(KANACHAN_WITH_UBSAN ON CACHE BOOL "With UndefinedBehaviorSanitizer.")
  endif()
endif()
if (KANACHAN_WITH_UBSAN)
  add_compile_options(-fsanitize=undefined)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=undefined")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
endif()
message("KANACHAN_WITH_UBSAN=${KANACHAN_WITH_UBSAN}")

if (KANACHAN_WITH_TSAN)
  add_compile_options(-fsanitize=thread)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
endif()
message("KANACHAN_WITH_TSAN=${KANACHAN_WITH_TSAN}")

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

add_compile_options(-pthread)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

message("CMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(Boost_USE_DEBUG_LIBS ON)
  set(Boost_USE_RELEASE_LIBS OFF)
else()
  set(Boost_USE_DEBUG_LIBS OFF)
  set(Boost_USE_RELEASE_LIBS ON)
endif()

set(BOOST_ROOT /usr/local)
find_package(Boost
  REQUIRED
  COMPONENTS stacktrace_backtrace python)

add_compile_definitions(BOOST_STACKTRACE_USE_BACKTRACE)

add_subdirectory(src)
add_subdirectory(test/annotation_vs_simulation)
