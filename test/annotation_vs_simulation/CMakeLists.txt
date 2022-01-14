add_executable(generate
  generate.cpp)
target_link_libraries(generate
  PRIVATE common
  PRIVATE Boost::headers)
