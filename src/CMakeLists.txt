add_library(kanachan
  post_zimo_action.cpp
  player_state.cpp
  utility.cpp
  throw.cpp
  type_name.cpp
  mahjongsoul.pb.cc)
target_link_libraries(kanachan
  protobuf
  Boost::stacktrace_backtrace
  Boost::headers)

add_executable(convert
  convert.cpp)
target_link_libraries(convert
  PRIVATE kanachan)
