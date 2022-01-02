include_directories(
  /home/ubuntu/.local/include
  /home/ubuntu/.local/src/shanten-number)
link_directories(/home/ubuntu/.local/lib)

add_executable(make_trie
  /home/ubuntu/.local/src/shanten-number/calsht.cpp
  make_trie.cpp)
target_link_libraries(make_trie
  PRIVATE marisa
  PRIVATE common)
