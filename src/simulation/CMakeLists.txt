set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

add_library(xiangting_calculator SHARED
  xiangting_calculator.cpp)
target_link_libraries(xiangting_calculator
  PRIVATE common
  PRIVATE marisa
  PRIVATE Boost::python
  PRIVATE Boost::headers
  PRIVATE "python${PYTHON_VERSION}"
  PRIVATE util
  PRIVATE dl)

add_library(simulation SHARED
  simulate.cpp
  simulator.cpp
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
  decision_maker.cpp
  gil.cpp
  utility.cpp)
target_link_libraries(simulation
  PRIVATE common
  PRIVATE marisa
  PRIVATE Boost::python
  PRIVATE Boost::headers
  PRIVATE "python${PYTHON_VERSION}"
  PRIVATE util
  PRIVATE dl)
