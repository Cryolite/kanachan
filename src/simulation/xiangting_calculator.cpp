#include "simulation/xiangting_calculator.hpp"

#include "common/throw.hpp"
#include <marisa.h>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/long.hpp>
#include <boost/python/object.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <ios>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

namespace{

using std::placeholders::_1;

} // namespace *unnamed*

class XiangtingCalculator::Impl_
{
public:
  explicit Impl_(std::filesystem::path const &prefix)
    : shupai_trie_(),
      shupai_xiangting_(),
      zipai_trie_(),
      zipai_xiangting_()
  {
    shupai_trie_.load((prefix / "shupai.trie").c_str());
    {
      std::ifstream ifs(
        prefix / "shupai.xiangting", std::ios_base::in | std::ios_base::binary);
      std::ostringstream oss;
      oss << ifs.rdbuf();
      std::string str = std::move(oss).str();
      if (str.size() != shupai_trie_.size() * 2u) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << str.size() << " != " << shupai_trie_.size() * 2u;
      }
      for (std::size_t i = 0u; i < str.size() / 2u; ++i) {
        shupai_xiangting_.emplace_back(str[i * 2u], str[i * 2u + 1u]);
      }
    }

    zipai_trie_.load((prefix / "zipai.trie").c_str());
    {
      std::ifstream ifs(
        prefix / "zipai.xiangting", std::ios_base::in | std::ios_base::binary);
      std::ostringstream oss;
      oss << ifs.rdbuf();
      std::string str = std::move(oss).str();
      if (str.size() != zipai_trie_.size() * 2u) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << str.size() << " != " << zipai_trie_.size() * 2u;
      }
      for (std::size_t i = 0u; i < str.size() / 2u; ++i) {
        zipai_xiangting_.emplace_back(str[i * 2u], str[i * 2u + 1u]);
      }
    }
  }

  explicit Impl_(std::string const &prefix)
    : Impl_(std::filesystem::path(prefix))
  {}

private:
  std::uint_fast8_t shupaiImpl_(
    std::string const &key, bool const headless) const
  {
    marisa::Agent agent;
    agent.set_query(key);
    if (!shupai_trie_.lookup(agent)) {
      std::ostringstream oss;
      oss << static_cast<unsigned>(key.front());
      for (char const c : key) {
        oss << ", " << static_cast<unsigned>(c);
      }
      KANACHAN_THROW<std::runtime_error>(_1) << std::move(oss).str();
    }
    std::size_t const id = agent.key().id();
    auto const xiangting = shupai_xiangting_[id];
    return headless ? xiangting.second : xiangting.first;
  }

  std::uint_fast8_t zipaiImpl_(
    std::string const &key, bool const headless) const
  {
    marisa::Agent agent;
    agent.set_query(key);
    if (!zipai_trie_.lookup(agent)) {
      std::ostringstream oss;
      oss << static_cast<unsigned>(key.front());
      for (char const c : key) {
        oss << ", " << static_cast<unsigned>(c);
      }
      KANACHAN_THROW<std::runtime_error>(_1) << std::move(oss).str();
    }
    std::size_t const id = agent.key().id();
    auto const xiangting = zipai_xiangting_[id];
    return headless ? xiangting.second : xiangting.first;
  }

  template<typename RandomAccessIterator>
  std::uint_fast8_t qiduiziImpl_(RandomAccessIterator first) const
  {
    std::uint_fast8_t num_pairs = 0u;
    for (std::uint_fast8_t i = 0; i < 34; ++i) {
      if (first[i] == 2u) {
        ++num_pairs;
      }
    }
    return 7u - num_pairs;
  }

  template<typename RandomAccessIterator>
  std::uint_fast8_t shisanyaoImpl_(RandomAccessIterator first) const
  {
    std::uint_fast8_t count = 0u;
    bool head = false;
    if (first[0u] >= 1u) {
      ++count;
      if (first[0u] >= 2u) {
        head = true;
      }
    }
    if (first[8u] >= 1u) {
      ++count;
      if (first[8u] >= 2u) {
        head = true;
      }
    }
    if (first[9u] >= 1u) {
      ++count;
      if (first[9u] >= 2u) {
        head = true;
      }
    }
    if (first[17u] >= 1u) {
      ++count;
      if (first[17u] >= 2u) {
        head = true;
      }
    }
    if (first[18u] >= 1u) {
      ++count;
      if (first[18u] >= 2u) {
        head = true;
      }
    }
    for (std::uint_fast8_t i = 26u; i < 34u; ++i) {
      if (first[i] >= 1u) {
        ++count;
        if (first[i] >= 2u) {
          head = true;
        }
      }
    }

    if (head) {
      ++count;
    }

    return 14u - count;
  }

public:
  template<typename RandomAccessIterator>
  std::uint_fast8_t operator()(
    RandomAccessIterator first, RandomAccessIterator last,
    std::uint_fast8_t const n) const
  {
    if (last - first != 34) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "(length of tile vector) = " << last - first;
    }
    {
      std::uint_fast16_t num_tiles = 0u;
      for (std::uint_fast8_t i = 0u; i < 34u; ++i) {
        if (first[i] < 0) {
          KANACHAN_THROW<std::invalid_argument>(_1)
            << "(# of " << i << ") = " << static_cast<int>(first[i]);
        }
        if (first[i] > 4u) {
          KANACHAN_THROW<std::invalid_argument>(_1)
            << "(# of " << i << ") = " << static_cast<unsigned>(first[i]);
        }
        num_tiles += first[i];
      }
      if (num_tiles > 3u * n + 2u) {
        KANACHAN_THROW<std::invalid_argument>(_1) << "(# of tiles) = " << num_tiles;
      }
    }

    std::uint_fast8_t xiangting = -1;
    {
      std::string key;
      std::array<std::uint_fast8_t, 5u> prev_headless_state{
        0u, 128u, 128u, 128u, 128u
      };
      std::array<std::uint_fast8_t, 5u> next_headless_state{
        static_cast<std::uint_fast8_t>(-1),
        static_cast<std::uint_fast8_t>(-1),
        static_cast<std::uint_fast8_t>(-1),
        static_cast<std::uint_fast8_t>(-1),
        static_cast<std::uint_fast8_t>(-1)
      };
      std::array<std::uint_fast8_t, 5u> prev_head_state{
        128u, 128u, 128u, 128u, 128u
      };
      std::array<std::uint_fast8_t, 5u> next_head_state{
        static_cast<std::uint_fast8_t>(-1),
        static_cast<std::uint_fast8_t>(-1),
        static_cast<std::uint_fast8_t>(-1),
        static_cast<std::uint_fast8_t>(-1),
        static_cast<std::uint_fast8_t>(-1)
      };
      for (std::uint_fast8_t i = 0u; i < 3u; ++i) {
        for (std::uint_fast8_t j = 0u; j <= 4u; ++j) {
          // j: 面子の数．
          key.clear();
          key.push_back(j);
          key.append(first + 9 * i, first + 9 * (i + 1));

          std::uint_fast8_t const headless_x = shupaiImpl_(key, true);
          std::uint_fast8_t const head_x = shupaiImpl_(key, false);

          for (std::uint_fast8_t k = 0u; k + j <= 4u; ++k) {
            std::uint_fast8_t const headless_xx
              = prev_headless_state[k] + headless_x;
            next_headless_state[k + j]
              = std::min(next_headless_state[k + j], headless_xx);
            std::uint_fast8_t const head_xx = prev_head_state[k] + headless_x;
            next_head_state[k + j] = std::min(next_head_state[k + j], head_xx);
          }
          for (std::uint_fast8_t k = 0u; k + j <= 4u; ++k) {
            std::uint_fast8_t const head_xx = prev_headless_state[k] + head_x;
            next_head_state[k + j] = std::min(next_head_state[k + j], head_xx);
          }
        }
        std::copy(
          next_headless_state.cbegin(), next_headless_state.cend(),
          prev_headless_state.begin());
        std::copy(
          next_head_state.cbegin(), next_head_state.cend(),
          prev_head_state.begin());
        std::fill(
          next_headless_state.begin(), next_headless_state.end(),
          static_cast<std::uint_fast8_t>(-1));
        std::fill(
          next_head_state.begin(), next_head_state.end(),
          static_cast<std::uint_fast8_t>(-1));
      }
      for (std::uint_fast8_t j = 0u; j <= 4u; ++j) {
        // j: 面子の数．
        key.clear();
        key.push_back(j);
        key.append(first + 27, last);

        std::uint_fast8_t const headless_x = zipaiImpl_(key, true);
        std::uint_fast8_t const head_x = zipaiImpl_(key, false);

        for (std::uint_fast8_t k = 0u; k + j <= 4u; ++k) {
          std::uint_fast8_t const headless_xx
            = prev_headless_state[k] + headless_x;
          next_headless_state[k + j]
            = std::min(next_headless_state[k + j], headless_xx);
          std::uint_fast8_t const head_xx = prev_head_state[k] + headless_x;
          next_head_state[k + j] = std::min(next_head_state[k + j], head_xx);
        }
        for (std::uint_fast8_t k = 0u; k + j <= 4u; ++k) {
          std::uint_fast8_t const head_xx = prev_headless_state[k] + head_x;
          next_head_state[k + j] = std::min(next_head_state[k + j], head_xx);
        }
      }
      xiangting = next_head_state[n];
    }

    {
      std::uint_fast8_t const x = qiduiziImpl_(first);
      xiangting = std::min(xiangting, x);
    }

    {
      std::uint_fast8_t const x = shisanyaoImpl_(first);
      xiangting = std::min(xiangting, x);
    }

    return xiangting;
  }

  template<typename RandomAccessRange>
  std::uint_fast8_t operator()(
    RandomAccessRange const &tiles, std::uint_fast8_t const n) const
  {
    return (*this)(cbegin(tiles), cend(tiles), n);
  }

private:
  marisa::Trie shupai_trie_;
  std::vector<std::pair<std::uint_fast8_t, std::uint_fast8_t> > shupai_xiangting_;
  marisa::Trie zipai_trie_;
  std::vector<std::pair<std::uint_fast8_t, std::uint_fast8_t> > zipai_xiangting_;
}; // class XiangtingCalculator::Impl_

XiangtingCalculator::XiangtingCalculator(std::filesystem::path const &prefix)
  : p_impl_(std::make_shared<Impl_>(prefix))
{}

XiangtingCalculator::XiangtingCalculator(std::string const &prefix)
  : XiangtingCalculator(std::filesystem::path(prefix))
{}

template<typename RandomAccessIterator>
std::uint_fast8_t XiangtingCalculator::operator()(
  RandomAccessIterator first, RandomAccessIterator last, std::uint_fast8_t n) const
{
  return (*p_impl_)(first, last, n);
}

template<typename RandomAccessRange>
std::uint_fast8_t XiangtingCalculator::operator()(
  RandomAccessRange const &tiles, std::uint_fast8_t n) const
{
  return (*p_impl_)(begin(tiles), end(tiles), n);
}

boost::python::long_ XiangtingCalculator::calculate(
  boost::python::list tile_counts, boost::python::long_ n) const
{
  long const length = boost::python::len(tile_counts);
  if (length != 34) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << "(length of tile vector) = " << length;
  }

  std::vector<std::uint_fast8_t> tile_counts_(34u, 0u);
  for (long tile = 0; tile < length; ++tile) {
    boost::python::object o = tile_counts[tile];
    long count = boost::python::extract<long>(o)();
    if (count < 0 || 4 < count) {
      KANACHAN_THROW<std::invalid_argument>(_1) << '(' << tile << ", " << count << ')';
    }
    tile_counts_[tile] = count;
  }

  long const n_ = boost::python::extract<long>(n)();
  if (n_ < 0 || 4 < n_) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "n = " << n_;
  }

  std::uint_fast8_t const result = (*p_impl_)(
    tile_counts_.cbegin(), tile_counts_.cend(), static_cast<std::uint_fast8_t>(n_));
  return boost::python::long_(result);
}

template<typename RandomAccessRange>
std::uint_fast8_t calculateXiangting(
  RandomAccessRange const &tiles, std::uint_fast8_t const n)
{
  static XiangtingCalculator xiangting_calculator;
  std::uint_fast8_t const xiangting = xiangting_calculator(tiles, n);
  return xiangting;
}

template
std::uint_fast8_t calculateXiangting<std::vector<std::uint_fast8_t> >(
  std::vector<std::uint_fast8_t> const &tiles, std::uint_fast8_t const n);

} // namespace Kanachan

BOOST_PYTHON_MODULE(_xiangting_calculator)
{
  boost::python::class_<Kanachan::XiangtingCalculator>(
    "XiangtingCalculator", boost::python::init<std::string>())
    .def("calculate", &Kanachan::XiangtingCalculator::calculate);
}
