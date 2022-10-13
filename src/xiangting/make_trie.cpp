#include "common/throw.hpp"
#include <calsht.hpp>
#include <marisa.h>
#include <fstream>
#include <ostream>
#include <ios>
#include <algorithm>
#include <vector>
#include <string_view>
#include <string>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <cstddef>


namespace{

void enumerate_shupai(
  std::uint_fast8_t const m, std::string &tiles, std::uint_fast8_t const i,
  std::uint_fast8_t const n, marisa::Keyset &keyset)
{
  std::string key;
  for (std::uint_fast8_t j = 0u; j <= std::min<std::uint_fast8_t>(n, 4u); ++j) {
    tiles[i] = j;
    if (i == 8u) {
      key.clear();
      key.push_back(m);
      key.append(tiles);
      keyset.push_back(key);
    }
    if (i < 8u) {
      enumerate_shupai(m, tiles, i + 1u, n - j, keyset);
    }
    tiles[i] = 0;
  }
}

void enumerate_zipai(
  std::uint_fast8_t const m, std::string &tiles, std::uint_fast8_t const i,
  std::uint_fast8_t const n, marisa::Keyset &keyset)
{
  std::string key;
  for (std::uint_fast8_t j = 0u; j <= std::min<std::uint_fast8_t>(n, 4u); ++j) {
    tiles[i] = j;
    if (i == 6u) {
      key.clear();
      key.push_back(m);
      key.append(tiles);
      keyset.push_back(key);
    }
    if (i < 6u) {
      enumerate_zipai(m, tiles, i + 1u, n - j, keyset);
    }
    tiles[i] = 0;
  }
}

} // namespace *unnamed*

int main(int argc, char const * const *argv)
{
  std::string input_prefix;
  std::string output_prefix;
  if (argc == 1) {
    KANACHAN_THROW<std::runtime_error>("error: Too few arguments.");
  }
  else if (argc == 2) {
    input_prefix = argv[1];
    output_prefix = input_prefix;
  }
  else if (argc == 3) {
    input_prefix = argv[1];
    output_prefix = argv[2];
  }
  else {
    KANACHAN_THROW<std::runtime_error>("error: Too many arguments.");
  }

  Calsht calsht;
  calsht.initialize(input_prefix.c_str());

  {
    marisa::Keyset shupai_keyset;
    {
      std::string tiles(9u, '\0');
      for (std::uint_fast8_t m = 0; m <= 4; ++m) {
        enumerate_shupai(m, tiles, 0u, 14u, shupai_keyset);
      }
    }
    marisa::Trie shupai_trie;
    shupai_trie.build(shupai_keyset);
    shupai_trie.save((output_prefix + "/shupai.trie").c_str());
    std::vector<std::pair<int, int> > shupai_xiangting(
      shupai_keyset.size(),
      std::pair(std::numeric_limits<int>::max(), std::numeric_limits<int>::max()));
    {
      std::vector<int> tiles(36u, 0);
      for (std::size_t i = 0u; i < shupai_keyset.size(); ++i) {
        marisa::Key const &key = shupai_keyset[i];
        std::string_view str = key.str();
        if (str.size() != 10u) {
          KANACHAN_THROW<std::logic_error>("");
        }
        std::copy(str.cbegin() + 1, str.cend(), tiles.begin());
        std::uint_fast8_t const xiangting
          = std::get<0u>(calsht(tiles, str.front(), 1));
        tiles[9u] = 2;
        std::uint_fast8_t const xiangting_headless
          = std::get<0u>(calsht(tiles, str.front(), 1));
        tiles[9u] = 0;
        std::size_t id = key.id();
        shupai_xiangting[id] = std::pair<int, int>(xiangting, xiangting_headless);
      }
    }
    std::ofstream ofs(
      output_prefix + "/shupai.xiangting", std::ios_base::out | std::ios_base::binary);
    for (auto const [xiangting, xiangting_headless] : shupai_xiangting) {
      if (xiangting == std::numeric_limits<int>::max()) {
        KANACHAN_THROW<std::logic_error>("");
      }
      if (xiangting_headless == std::numeric_limits<int>::max()) {
        KANACHAN_THROW<std::logic_error>("");
      }
      ofs << static_cast<char>(xiangting)
          << static_cast<char>(xiangting_headless);
    }
    ofs << std::flush;
  }

  {
    marisa::Keyset zipai_keyset;
    {
      std::string tiles(7u, '\0');
      for (std::uint_fast8_t m = 0; m <= 4; ++m) {
        enumerate_zipai(m, tiles, 0u, 14u, zipai_keyset);
      }
    }
    marisa::Trie zipai_trie;
    zipai_trie.build(zipai_keyset);
    zipai_trie.save((output_prefix + "/zipai.trie").c_str());
    std::vector<std::pair<int, int> > zipai_xiangting(
      zipai_keyset.size(),
      std::pair(std::numeric_limits<int>::max(), std::numeric_limits<int>::max()));
    {
      std::vector<int> tiles(36u, 0);
      for (std::size_t i = 0u; i < zipai_keyset.size(); ++i) {
        marisa::Key const &key = zipai_keyset[i];
        std::string_view str = key.str();
        if (str.size() != 8u) {
          KANACHAN_THROW<std::logic_error>("");
        }
        std::copy(str.cbegin() + 1, str.cend(), tiles.begin() + 27);
        std::uint_fast8_t const xiangting
          = std::get<0u>(calsht(tiles, str.front(), 1));
        tiles[0u] = 2;
        std::uint_fast8_t const xiangting_headless
          = std::get<0u>(calsht(tiles, str.front(), 1));
        tiles[0u] = 0;
        std::size_t id = key.id();
        zipai_xiangting[id] = std::pair<int, int>(xiangting, xiangting_headless);
      }
    }
    std::ofstream ofs(
      output_prefix + "/zipai.xiangting", std::ios_base::out | std::ios_base::binary);
    for (auto const [xiangting, xiangting_headless] : zipai_xiangting) {
      if (xiangting == std::numeric_limits<int>::max()) {
        KANACHAN_THROW<std::logic_error>("");
      }
      if (xiangting_headless == std::numeric_limits<int>::max()) {
        KANACHAN_THROW<std::logic_error>("");
      }
      ofs << static_cast<char>(xiangting)
          << static_cast<char>(xiangting_headless);
    }
    ofs << std::flush;
  }
}
