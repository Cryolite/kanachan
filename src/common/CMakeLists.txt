add_library(common
  mahjongsoul.pb.cc
  assert.cpp
  throw.cpp
  type_name.cpp)
target_link_libraries(common
  PRIVATE protobuf
  PRIVATE Boost::stacktrace_backtrace
  PRIVATE Boost::headers)
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE ON)
