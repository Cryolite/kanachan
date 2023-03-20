if (DEFINED SHANTEN_NUMBER_SOURCE_PATH)
  include_directories("${SHANTEN_NUMBER_SOURCE_PATH}")

  add_executable(make_trie
    "${SHANTEN_NUMBER_SOURCE_PATH}/calsht.cpp"
    make_trie.cpp)
  target_link_libraries(make_trie
    PRIVATE dl
    PRIVATE marisa
    PRIVATE common)
endif()
