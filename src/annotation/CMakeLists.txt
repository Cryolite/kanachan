add_executable(annotate
  main.cpp
  annotation.cpp
  round_progress.cpp
  player_state.cpp
  utility.cpp)
target_link_libraries(annotate
  PRIVATE common
  PRIVATE Boost::headers)
