include_directories(/home/ubuntu/.local/include)
link_directories(/home/ubuntu/.local/lib)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

add_library(_simulation SHARED
  simulate.cpp
  duplicated_games.cpp
  game.cpp
  round.cpp
  sijia_lizhi.cpp
  sigang_sanle.cpp
  sifeng_lianda.cpp
  jiuzhong_jiupai.cpp
  huangpai_pingju.cpp
  hule.cpp
  daminggang.cpp
  peng.cpp
  chi.cpp
  dapai.cpp
  jiagang.cpp
  angang.cpp
  zimo.cpp
  round_state.cpp
  shoupai.cpp
  xiangting_calculator.cpp
  paishan.cpp
  game_state.cpp
  utility.cpp)
target_link_libraries(_simulation
  PRIVATE common
  PRIVATE marisa
  PRIVATE Boost::python
  PRIVATE Boost::headers
  PRIVATE "python${PYTHON_VERSION}")
