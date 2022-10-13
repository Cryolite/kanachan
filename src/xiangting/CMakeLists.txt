if (DEFINED SHANTEN_NUMBER_SOURCE_PATH)
  include_directories(
    "${MARISA_TRIE_ROOT}/include"
    "${SHANTEN_NUMBER_SOURCE_PATH}")
  link_directories("${MARISA_TRIE_ROOT}/lib")

  add_executable(make_trie
    "${SHANTEN_NUMBER_SOURCE_PATH}/calsht.cpp"
    make_trie.cpp)
  target_link_libraries(make_trie
    PRIVATE marisa
    PRIVATE common)
endif()
