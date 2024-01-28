#define PY_SSIZE_T_CLEAN
#include "simulation/shoupai.hpp"

#include "simulation/xiangting_calculator.hpp"
#include "simulation/paishan.hpp"
#include "simulation/gil.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/import.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/str.hpp>
#include <boost/python/object.hpp>
#include <Python.h>
#include <numeric>
#include <algorithm>
#include <vector>
#include <array>
#include <functional>
#include <utility>
#include <stdexcept>
#include <limits>
#include <cstdint>


namespace Kanachan{

namespace{

using std::placeholders::_1;
namespace python = boost::python;

constexpr std::array<std::uint_fast8_t, 37u> offsets_to_136_{
    0u,   1u,   5u,   9u,  13u,  17u,  20u,  24u,  28u,  32u,
   36u,  37u,  41u,  45u,  49u,  53u,  56u,  60u,  64u,  68u,
   72u,  73u,  77u,  81u,  85u,  89u,  92u,  96u, 100u, 104u,
  108u, 112u, 116u, 120u, 124u, 128u, 132u
};

constexpr std::array<std::array<std::uint_fast8_t, 4u>, 90u> chi_encode_ = {
  std::array<std::uint_fast8_t, 4u>{ 2u, 3u, 1u, 4u },
  std::array<std::uint_fast8_t, 4u>{ 1u, 3u, 2u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 3u, 4u, 2u, 5u },
  std::array<std::uint_fast8_t, 4u>{ 1u, 2u, 3u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 2u, 4u, 3u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 4u, 5u, 3u, 6u },
  std::array<std::uint_fast8_t, 4u>{ 4u, 0u, 3u, 6u },
  std::array<std::uint_fast8_t, 4u>{ 2u, 3u, 4u, 1u },
  std::array<std::uint_fast8_t, 4u>{ 3u, 5u, 4u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 3u, 0u, 4u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 5u, 6u, 4u, 7u },
  std::array<std::uint_fast8_t, 4u>{ 0u, 6u, 4u, 7u },
  std::array<std::uint_fast8_t, 4u>{ 3u, 4u, 5u, 2u },
  std::array<std::uint_fast8_t, 4u>{ 3u, 4u, 0u, 2u },
  std::array<std::uint_fast8_t, 4u>{ 4u, 6u, 5u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 4u, 6u, 0u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 6u, 7u, 5u, 8u },
  std::array<std::uint_fast8_t, 4u>{ 6u, 7u, 0u, 8u },
  std::array<std::uint_fast8_t, 4u>{ 4u, 5u, 6u, 3u },
  std::array<std::uint_fast8_t, 4u>{ 4u, 0u, 6u, 3u },
  std::array<std::uint_fast8_t, 4u>{ 5u, 7u, 6u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 0u, 7u, 6u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 7u, 8u, 6u, 9u },
  std::array<std::uint_fast8_t, 4u>{ 5u, 6u, 7u, 4u },
  std::array<std::uint_fast8_t, 4u>{ 0u, 6u, 7u, 4u },
  std::array<std::uint_fast8_t, 4u>{ 6u, 8u, 7u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 8u, 9u, 7u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 6u, 7u, 8u, 5u },
  std::array<std::uint_fast8_t, 4u>{ 7u, 9u, 8u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 7u, 8u, 9u, 6u },
  std::array<std::uint_fast8_t, 4u>{ 12u, 13u, 11u, 14u },
  std::array<std::uint_fast8_t, 4u>{ 11u, 13u, 12u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 13u, 14u, 12u, 15u },
  std::array<std::uint_fast8_t, 4u>{ 11u, 12u, 13u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 12u, 14u, 13u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 14u, 15u, 13u, 16u },
  std::array<std::uint_fast8_t, 4u>{ 14u, 10u, 13u, 16u },
  std::array<std::uint_fast8_t, 4u>{ 12u, 13u, 14u, 11u },
  std::array<std::uint_fast8_t, 4u>{ 13u, 15u, 14u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 13u, 10u, 14u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 15u, 16u, 14u, 17u },
  std::array<std::uint_fast8_t, 4u>{ 10u, 16u, 14u, 17u },
  std::array<std::uint_fast8_t, 4u>{ 13u, 14u, 15u, 12u },
  std::array<std::uint_fast8_t, 4u>{ 13u, 14u, 10u, 12u },
  std::array<std::uint_fast8_t, 4u>{ 14u, 16u, 15u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 14u, 16u, 10u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 16u, 17u, 15u, 18u },
  std::array<std::uint_fast8_t, 4u>{ 16u, 17u, 10u, 18u },
  std::array<std::uint_fast8_t, 4u>{ 14u, 15u, 16u, 13u },
  std::array<std::uint_fast8_t, 4u>{ 14u, 10u, 16u, 13u },
  std::array<std::uint_fast8_t, 4u>{ 15u, 17u, 16u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 10u, 17u, 16u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 17u, 18u, 16u, 19u },
  std::array<std::uint_fast8_t, 4u>{ 15u, 16u, 17u, 14u },
  std::array<std::uint_fast8_t, 4u>{ 10u, 16u, 17u, 14u },
  std::array<std::uint_fast8_t, 4u>{ 16u, 18u, 17u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 18u, 19u, 17u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 16u, 17u, 18u, 15u },
  std::array<std::uint_fast8_t, 4u>{ 17u, 19u, 18u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 17u, 18u, 19u, 16u },
  std::array<std::uint_fast8_t, 4u>{ 22u, 23u, 21u, 24u },
  std::array<std::uint_fast8_t, 4u>{ 21u, 23u, 22u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 23u, 24u, 22u, 25u },
  std::array<std::uint_fast8_t, 4u>{ 21u, 22u, 23u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 22u, 24u, 23u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 24u, 25u, 23u, 26u },
  std::array<std::uint_fast8_t, 4u>{ 24u, 20u, 23u, 26u },
  std::array<std::uint_fast8_t, 4u>{ 22u, 23u, 24u, 21u },
  std::array<std::uint_fast8_t, 4u>{ 23u, 25u, 24u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 23u, 20u, 24u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 25u, 26u, 24u, 27u },
  std::array<std::uint_fast8_t, 4u>{ 20u, 26u, 24u, 27u },
  std::array<std::uint_fast8_t, 4u>{ 23u, 24u, 25u, 22u },
  std::array<std::uint_fast8_t, 4u>{ 23u, 24u, 20u, 22u },
  std::array<std::uint_fast8_t, 4u>{ 24u, 26u, 25u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 24u, 26u, 20u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 26u, 27u, 25u, 28u },
  std::array<std::uint_fast8_t, 4u>{ 26u, 27u, 20u, 28u },
  std::array<std::uint_fast8_t, 4u>{ 24u, 25u, 26u, 23u },
  std::array<std::uint_fast8_t, 4u>{ 24u, 20u, 26u, 23u },
  std::array<std::uint_fast8_t, 4u>{ 25u, 27u, 26u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 20u, 27u, 26u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 27u, 28u, 26u, 29u },
  std::array<std::uint_fast8_t, 4u>{ 25u, 26u, 27u, 24u },
  std::array<std::uint_fast8_t, 4u>{ 20u, 26u, 27u, 24u },
  std::array<std::uint_fast8_t, 4u>{ 26u, 28u, 27u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 28u, 29u, 27u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 26u, 27u, 28u, 25u },
  std::array<std::uint_fast8_t, 4u>{ 27u, 29u, 28u, std::numeric_limits<std::uint_fast8_t>::max() },
  std::array<std::uint_fast8_t, 4u>{ 27u, 28u, 29u, 26u }
};

constexpr std::array<std::array<std::uint_fast8_t, 3u>, 40u> peng_encode_ = {
  std::array<std::uint_fast8_t, 3u>{ 1u, 1u, 1u },    //  0
  std::array<std::uint_fast8_t, 3u>{ 2u, 2u, 2u },    //  1
  std::array<std::uint_fast8_t, 3u>{ 3u, 3u, 3u },    //  2
  std::array<std::uint_fast8_t, 3u>{ 4u, 4u, 4u },    //  3
  std::array<std::uint_fast8_t, 3u>{ 5u, 5u, 5u },    //  4
  std::array<std::uint_fast8_t, 3u>{ 0u, 5u, 5u },    //  5
  std::array<std::uint_fast8_t, 3u>{ 5u, 5u, 0u },    //  6
  std::array<std::uint_fast8_t, 3u>{ 6u, 6u, 6u },    //  7
  std::array<std::uint_fast8_t, 3u>{ 7u, 7u, 7u },    //  8
  std::array<std::uint_fast8_t, 3u>{ 8u, 8u, 8u },    //  9
  std::array<std::uint_fast8_t, 3u>{ 9u, 9u, 9u },    // 10
  std::array<std::uint_fast8_t, 3u>{ 11u, 11u, 11u }, // 11
  std::array<std::uint_fast8_t, 3u>{ 12u, 12u, 12u }, // 12
  std::array<std::uint_fast8_t, 3u>{ 13u, 13u, 13u }, // 13
  std::array<std::uint_fast8_t, 3u>{ 14u, 14u, 14u }, // 14
  std::array<std::uint_fast8_t, 3u>{ 15u, 15u, 15u }, // 15
  std::array<std::uint_fast8_t, 3u>{ 10u, 15u, 15u }, // 16
  std::array<std::uint_fast8_t, 3u>{ 15u, 15u, 10u }, // 17
  std::array<std::uint_fast8_t, 3u>{ 16u, 16u, 16u }, // 18
  std::array<std::uint_fast8_t, 3u>{ 17u, 17u, 17u }, // 19
  std::array<std::uint_fast8_t, 3u>{ 18u, 18u, 18u }, // 20
  std::array<std::uint_fast8_t, 3u>{ 19u, 19u, 19u }, // 21
  std::array<std::uint_fast8_t, 3u>{ 21u, 21u, 21u }, // 22
  std::array<std::uint_fast8_t, 3u>{ 22u, 22u, 22u }, // 23
  std::array<std::uint_fast8_t, 3u>{ 23u, 23u, 23u }, // 24
  std::array<std::uint_fast8_t, 3u>{ 24u, 24u, 24u }, // 25
  std::array<std::uint_fast8_t, 3u>{ 25u, 25u, 25u }, // 26
  std::array<std::uint_fast8_t, 3u>{ 20u, 25u, 25u }, // 27
  std::array<std::uint_fast8_t, 3u>{ 25u, 25u, 20u }, // 28
  std::array<std::uint_fast8_t, 3u>{ 26u, 26u, 26u }, // 29
  std::array<std::uint_fast8_t, 3u>{ 27u, 27u, 27u }, // 30
  std::array<std::uint_fast8_t, 3u>{ 28u, 28u, 28u }, // 31
  std::array<std::uint_fast8_t, 3u>{ 29u, 29u, 29u }, // 32
  std::array<std::uint_fast8_t, 3u>{ 30u, 30u, 30u }, // 33
  std::array<std::uint_fast8_t, 3u>{ 31u, 31u, 31u }, // 34
  std::array<std::uint_fast8_t, 3u>{ 32u, 32u, 32u }, // 35
  std::array<std::uint_fast8_t, 3u>{ 33u, 33u, 33u }, // 36
  std::array<std::uint_fast8_t, 3u>{ 34u, 34u, 34u }, // 37
  std::array<std::uint_fast8_t, 3u>{ 35u, 35u, 35u }, // 38
  std::array<std::uint_fast8_t, 3u>{ 36u, 36u, 36u }  // 39
};

constexpr std::array<std::array<std::uint_fast8_t, 3u>, 37u> daminggang_encode_ = {
  std::array<std::uint_fast8_t, 3u>{ 5u, 5u, 5u },
  std::array<std::uint_fast8_t, 3u>{ 1u, 1u, 1u },
  std::array<std::uint_fast8_t, 3u>{ 2u, 2u, 2u },
  std::array<std::uint_fast8_t, 3u>{ 3u, 3u, 3u },
  std::array<std::uint_fast8_t, 3u>{ 4u, 4u, 4u },
  std::array<std::uint_fast8_t, 3u>{ 5u, 5u, 0u },
  std::array<std::uint_fast8_t, 3u>{ 6u, 6u, 6u },
  std::array<std::uint_fast8_t, 3u>{ 7u, 7u, 7u },
  std::array<std::uint_fast8_t, 3u>{ 8u, 8u, 8u },
  std::array<std::uint_fast8_t, 3u>{ 9u, 9u, 9u },
  std::array<std::uint_fast8_t, 3u>{ 15u, 15u, 15u },
  std::array<std::uint_fast8_t, 3u>{ 11u, 11u, 11u },
  std::array<std::uint_fast8_t, 3u>{ 12u, 12u, 12u },
  std::array<std::uint_fast8_t, 3u>{ 13u, 13u, 13u },
  std::array<std::uint_fast8_t, 3u>{ 14u, 14u, 14u },
  std::array<std::uint_fast8_t, 3u>{ 15u, 15u, 10u },
  std::array<std::uint_fast8_t, 3u>{ 16u, 16u, 16u },
  std::array<std::uint_fast8_t, 3u>{ 17u, 17u, 17u },
  std::array<std::uint_fast8_t, 3u>{ 18u, 18u, 18u },
  std::array<std::uint_fast8_t, 3u>{ 19u, 19u, 19u },
  std::array<std::uint_fast8_t, 3u>{ 25u, 25u, 25u },
  std::array<std::uint_fast8_t, 3u>{ 21u, 21u, 21u },
  std::array<std::uint_fast8_t, 3u>{ 22u, 22u, 22u },
  std::array<std::uint_fast8_t, 3u>{ 23u, 23u, 23u },
  std::array<std::uint_fast8_t, 3u>{ 24u, 24u, 24u },
  std::array<std::uint_fast8_t, 3u>{ 25u, 25u, 20u },
  std::array<std::uint_fast8_t, 3u>{ 26u, 26u, 26u },
  std::array<std::uint_fast8_t, 3u>{ 27u, 27u, 27u },
  std::array<std::uint_fast8_t, 3u>{ 28u, 28u, 28u },
  std::array<std::uint_fast8_t, 3u>{ 29u, 29u, 29u },
  std::array<std::uint_fast8_t, 3u>{ 30u, 30u, 30u },
  std::array<std::uint_fast8_t, 3u>{ 31u, 31u, 31u },
  std::array<std::uint_fast8_t, 3u>{ 32u, 32u, 32u },
  std::array<std::uint_fast8_t, 3u>{ 33u, 33u, 33u },
  std::array<std::uint_fast8_t, 3u>{ 34u, 34u, 34u },
  std::array<std::uint_fast8_t, 3u>{ 35u, 35u, 35u },
  std::array<std::uint_fast8_t, 3u>{ 36u, 36u, 36u }
};

} // namespace `anonymous`

void swap(Shoupai &lhs, Shoupai &rhs) noexcept
{
  lhs.swap(rhs);
}

Shoupai::Shoupai(std::uint_fast8_t const index, Kanachan::Paishan const &paishan)
  : external_tool_()
{
  KANACHAN_ASSERT((index < 4u));

  for (std::uint_fast8_t i = 0u; i < 3u; ++i) {
    for (std::uint_fast8_t j = 0u; j < 4u; ++j) {
      std::uint_fast8_t const tile = paishan[i * 16u + index * 4u + j];
      KANACHAN_ASSERT((tile < 37u));
      ++shoupai_[tile];
    }
  }
  std::uint_fast8_t const tile = paishan[48u + index];
  KANACHAN_ASSERT((tile < 37u));
  ++shoupai_[tile];

  {
    Kanachan::GIL::RecursiveLock gil_lock;
    python::object m = python::import("kanachan.simulation");
    external_tool_ = m.attr("Tool")();
  }

  if (isTingpai()) {
    updateHupaiList_();
  }
}

Shoupai::~Shoupai()
{
  Kanachan::GIL::RecursiveLock gil_lock;
  external_tool_ = python::object();
}

void Shoupai::swap(Shoupai &rhs) noexcept
{
  using std::swap;
  swap(shoupai_, rhs.shoupai_);
  swap(fulu_list_, rhs.fulu_list_);
}

Shoupai &Shoupai::operator=(Shoupai const &rhs)
{
  Shoupai(rhs).swap(*this);
  return *this;
}

Shoupai &Shoupai::operator=(Shoupai &&rhs) noexcept
{
  Shoupai(std::move(rhs)).swap(*this);
  return *this;
}

bool Shoupai::isMenqian() const
{
  for (std::uint_fast16_t const fulu : fulu_list_) {
    if (fulu == std::numeric_limits<std::uint_fast16_t>::max()) {
      return true;
    }
    if (fulu < 90u) {
      // チー
      return false;
    }
    if (/*90u <= fulu && */fulu < 210u) {
      // ポン
      return false;
    }
    if (/*210u <= fulu && */fulu < 321u) {
      // 大明槓
      return false;
    }
    if (/*321u <= fulu && */fulu < 355u) {
      // 暗槓
      continue;
    }
    if (/*355u <= fulu && */fulu < 392u) {
      return false;
    }
    KANACHAN_THROW<std::logic_error>(_1) << fulu << ": A logic error.";
  }
  return true;
}

bool Shoupai::isTingpai() const
{
  if (tingpai_cache_) {
    return true;
  }

  std::vector<std::uint_fast8_t> shoupai34 = getShoupai34_();
  std::uint_fast8_t const num_fulu = getNumFulu_();
  std::uint_fast8_t const xiangting = Kanachan::calculateXiangting(shoupai34, 4u - num_fulu);
  tingpai_cache_ = (xiangting == 1u);
  xiangting_lower_bound_ = xiangting;
  return tingpai_cache_;
}

std::uint_fast8_t Shoupai::getNumGangzi() const
{
  std::uint_fast8_t n = 0u;
  for (std::uint_fast16_t const fulu : fulu_list_) {
    if (fulu == std::numeric_limits<std::uint_fast16_t>::max()) {
      break;
    }
    if (fulu >= 210u) {
      KANACHAN_ASSERT((fulu < 392u));
      ++n;
    }
  }
  return n;
}

std::vector<std::uint_fast8_t> Shoupai::getShoupai34_() const
{
  std::vector<std::uint_fast8_t> shoupai34(34u, 0u);

  shoupai34[4u] += shoupai_[0u];
  for (std::uint_fast8_t i = 0u; i < 9u; ++i) {
    shoupai34[i] += shoupai_[i + 1u];
  }
  shoupai34[13u] += shoupai_[10u];
  for (std::uint_fast8_t i = 9u; i < 18u; ++i) {
    shoupai34[i] += shoupai_[i + 2u];
  }
  shoupai34[22u] += shoupai_[20u];
  for (std::uint_fast8_t i = 18u; i < 34u; ++i) {
    shoupai34[i] += shoupai_[i + 3u];
  }

  return shoupai34;
}

std::uint_fast8_t Shoupai::getNumFulu_() const
{
  std::uint_fast8_t n = 0u;
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    if (fulu_list_[i] == std::numeric_limits<std::uint_fast16_t>::max()) {
      return n;
    }
    ++n;
  }
  KANACHAN_ASSERT((n == 4u));
  return n;
}

python::list Shoupai::getShoupai136_(python::list fulu_list, std::uint_fast8_t const hupai) const
{
  if (PyGILState_Check() != 1) {
    KANACHAN_THROW<std::logic_error>("The Python GIL must be held.");
  }

  python::list shoupai136;

  auto convert = [](std::uint_fast8_t const tile) -> std::uint_fast8_t {
    if (tile == 0) {
      return 16;
    }
    else if (1 <= tile && tile <= 4) {
      return (tile - 1) * 4;
    }
    else if (tile == 5) {
      return 17;
    }
    else if (6 <= tile && tile <= 9) {
      return (tile - 1) * 4;
    }
    else if (tile == 10) {
      return 52;
    }
    else if (11 <= tile && tile <= 14) {
      return (tile - 2) * 4;
    }
    else if (tile == 15) {
      return 53;
    }
    else if (16 <= tile && tile <= 19) {
      return (tile - 2) * 4;
    }
    else if (tile == 20) {
      return 88;
    }
    else if (21 <= tile && tile <= 24) {
      return (tile - 3) * 4;
    }
    else if (tile == 25) {
      return 89;
    }
    else {
      KANACHAN_ASSERT((26 <= tile && tile < 37));
      return (tile - 3) * 4;
    }
    KANACHAN_THROW<std::logic_error>("");
  };

  for (std::uint_fast8_t i = 0u; i < 37u; ++i) {
    std::uint_fast8_t const offset = convert(i);
    std::uint_fast8_t const num_tiles = shoupai_[i] + (i == hupai ? 1u : 0u);
    for (std::uint_fast8_t j = 0u; j < num_tiles; ++j) {
      shoupai136.append(offset + j);
    }
  }

  // `shoupai136` は `fulu_list` の牌を含めなければならない．
  for (long i = 0; i < python::len(fulu_list); ++i) {
    python::object o = fulu_list[i];
    python::extract<python::tuple> t(o);
    o = t()[1];
    python::extract<python::list> l(o);
    for (long j = 0; j < 3; ++j) {
      o = l()[j];
      python::extract<long> tile(o);
      if (tile() == 16) {
        KANACHAN_ASSERT((shoupai136.count(16) == 0));
        shoupai136.append(16);
      }
      else if (tile() / 4 == 4) {
        bool flag = false;
        for (long k = tile(); k < tile() + 3; ++k) {
          if (shoupai136.count(k) == 0) {
            shoupai136.append(k);
            flag = true;
            break;
          }
        }
        KANACHAN_ASSERT((flag));
      }
      else if (tile() == 52) {
        KANACHAN_ASSERT((shoupai136.count(52) == 0));
        shoupai136.append(52);
      }
      else if (tile() / 4 == 13) {
        bool flag = false;
        for (long k = tile(); k < tile() + 3; ++k) {
          if (shoupai136.count(k) == 0) {
            shoupai136.append(k);
            flag = true;
            break;
          }
        }
        KANACHAN_ASSERT((flag));
      }
      else if (tile() == 88) {
        KANACHAN_ASSERT((shoupai136.count(88) == 0));
        shoupai136.append(88);
      }
      else if (tile() / 4 == 22) {
        bool flag = false;
        for (long k = tile(); k < tile() + 3; ++k) {
          if (shoupai136.count(k) == 0) {
            shoupai136.append(k);
            flag = true;
            break;
          }
        }
        KANACHAN_ASSERT((flag));
      }
      else {
        bool flag = false;
        for (long k = tile(); k < tile() + 4; ++k) {
          if (shoupai136.count(k) == 0) {
            shoupai136.append(k);
            flag = true;
            break;
          }
        }
        KANACHAN_ASSERT((flag));
      }
    }
  }

  shoupai136.attr("sort")();
  return shoupai136;
}

python::list Shoupai::getFuluList_() const
{
  if (PyGILState_Check() != 1) {
    KANACHAN_THROW<std::logic_error>("The Python GIL must be held.");
  }

  auto convert = [](std::uint_fast8_t const tile) -> std::uint_fast8_t {
    return tile;
  };

  python::list fulu_list;
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    std::uint_fast16_t const fulu = fulu_list_[i];
    if (fulu == std::numeric_limits<std::uint_fast16_t>::max()) {
      break;
    }

    if (fulu < 90u) {
      // Chi (チー)
      std::array<std::uint_fast8_t, 3u> tiles{
        chi_encode_[fulu][0u],
        chi_encode_[fulu][1u],
        chi_encode_[fulu][2u]
      };
      std::sort(tiles.begin(), tiles.end());
      python::list tiles_;
      for (std::uint_fast8_t j = 0u; j < tiles.size(); ++j) {
        std::uint_fast8_t const tile = tiles[j];
        if (tile == 0u) {
          tiles_.append(16);
        }
        else if (tile == 5u) {
          tiles_.append(17);
        }
        else if (tile == 10u) {
          tiles_.append(52);
        }
        else if (tile == 15u) {
          tiles_.append(53);
        }
        else if (tile == 20u) {
          tiles_.append(88);
        }
        else if (tile == 25u) {
          tiles_.append(89);
        }
        else if (1u <= tile && tile <= 4u || 6u <= tile && tile <= 9u) {
          tiles_.append((tile - 1) * 4);
        }
        else if (11u <= tile && tile <= 14u || 16u <= tile && tile <= 19u) {
          tiles_.append((tile - 2) * 4);
        }
        else if (21u <= tile && tile <= 24u || 26u <= tile && tile < 37u) {
          tiles_.append((tile - 3) * 4);
        }
      }
      tiles_.attr("sort")();
      python::tuple t = python::make_tuple(python::str("chi"), tiles_, 1);
      fulu_list.append(t);
    }
    else if (/*90u <= fulu && */fulu < 210u) {
      // Peng (ポン)
      std::uint_fast8_t const relseat = (fulu - 90u) / 40u;
      std::uint_fast8_t const encode = fulu - 90u - relseat * 40u;
      KANACHAN_ASSERT((encode < 40u));
      python::list tiles_;
      if (encode <= 3u) {
        tiles_.append(encode * 4);
        tiles_.append(encode * 4 + 1);
        tiles_.append(encode * 4 + 2);
      }
      else if (encode == 4u) {
        tiles_.append(17);
        tiles_.append(18);
        tiles_.append(19);
      }
      else if (encode == 5u || encode == 6u) {
        tiles_.append(16);
        tiles_.append(17);
        tiles_.append(18);
      }
      else if (7u <= encode && encode <= 14u) {
        tiles_.append((encode - 2) * 4);
        tiles_.append((encode - 2) * 4 + 1);
        tiles_.append((encode - 2) * 4 + 2);
      }
      else if (encode == 15u) {
        tiles_.append(53);
        tiles_.append(54);
        tiles_.append(55);
      }
      else if (encode == 16u || encode == 17u) {
        tiles_.append(52);
        tiles_.append(53);
        tiles_.append(54);
      }
      else if (18u <= encode && encode <= 25u) {
        tiles_.append((encode - 4) * 4);
        tiles_.append((encode - 4) * 4 + 1);
        tiles_.append((encode - 4) * 4 + 2);
      }
      else if (encode == 26u) {
        tiles_.append(89u);
        tiles_.append(90u);
        tiles_.append(91u);
      }
      else if (encode == 27u || encode == 28u) {
        tiles_.append(88u);
        tiles_.append(89u);
        tiles_.append(90u);
      }
      else {
        KANACHAN_ASSERT((29u <= encode && encode < 40u));
        tiles_.append((encode - 6) * 4);
        tiles_.append((encode - 6) * 4 + 1);
        tiles_.append((encode - 6) * 4 + 2);
      }
      python::tuple t = python::make_tuple(python::str("pon"), tiles_, 1);
      fulu_list.append(t);
    }
    else if (/*210u <= fulu && */fulu < 321u) {
      // Da Ming Gang (大明槓)
      std::uint_fast8_t const relseat = (fulu - 210u) / 37u;
      std::uint_fast8_t const encode = fulu - 210u - relseat * 37u;
      KANACHAN_ASSERT((encode < 37u));
      python::list tiles_;
      if (encode == 0u || encode == 5u) {
        tiles_.append(16);
        tiles_.append(17);
        tiles_.append(18);
        tiles_.append(19);
      }
      else if (encode == 10u || encode == 15u) {
        tiles_.append(52);
        tiles_.append(53);
        tiles_.append(54);
        tiles_.append(55);
      }
      else if (encode == 20u || encode == 25u) {
        tiles_.append(88);
        tiles_.append(89);
        tiles_.append(90);
        tiles_.append(91);
      }
      else if (1u <= encode && encode <= 4u || 6u <= encode && encode <= 9u) {
        tiles_.append((encode - 1) * 4);
        tiles_.append((encode - 1) * 4 + 1);
        tiles_.append((encode - 1) * 4 + 2);
        tiles_.append((encode - 1) * 4 + 3);
      }
      else if (11u <= encode && encode <= 14u || 16u <= encode && encode <= 19u) {
        tiles_.append((encode - 2) * 4);
        tiles_.append((encode - 2) * 4 + 1);
        tiles_.append((encode - 2) * 4 + 2);
        tiles_.append((encode - 2) * 4 + 3);
      }
      else if (21u <= encode && encode <= 24u || 26u <= encode && encode < 37u) {
        tiles_.append((encode - 3) * 4);
        tiles_.append((encode - 3) * 4 + 1);
        tiles_.append((encode - 3) * 4 + 2);
        tiles_.append((encode - 3) * 4 + 3);
      }
      python::tuple t = python::make_tuple(python::str("kan"), tiles_, 1);
      fulu_list.append(t);
    }
    else if (/*321u <= fulu && */fulu < 355u) {
      // An Gang (暗槓)
      std::uint_fast8_t const encode = fulu - 321u;
      python::list tiles_;
      if (encode == 4u || encode == 13u || encode == 22u) {
        tiles_.append(encode * 4);
        tiles_.append(encode * 4 + 1);
        tiles_.append(encode * 4 + 2);
        tiles_.append(encode * 4 + 3);
      }
      else {
        tiles_.append(encode * 4);
        tiles_.append(encode * 4 + 1);
        tiles_.append(encode * 4 + 2);
        tiles_.append(encode * 4 + 3);
      }
      python::tuple t = python::make_tuple(python::str("kan"), tiles_, 0);
      fulu_list.append(t);
    }
    else {
      // Jia Gang (加槓)
      KANACHAN_ASSERT((355u <= fulu && fulu < 392));
      std::uint_fast8_t const encode = fulu - 355u;
      KANACHAN_ASSERT((encode < 37u));
      python::list tiles_;
      if (encode == 0u || encode == 5u) {
        tiles_.append(16);
        tiles_.append(17);
        tiles_.append(18);
        tiles_.append(19);
      }
      else if (encode == 10u || encode == 15u) {
        tiles_.append(52);
        tiles_.append(53);
        tiles_.append(54);
        tiles_.append(55);
      }
      else if (encode == 20u || encode == 25u) {
        tiles_.append(88);
        tiles_.append(89);
        tiles_.append(90);
        tiles_.append(91);
      }
      else if (1u <= encode && encode <= 4u || 6u <= encode && encode <= 9u) {
        tiles_.append((encode - 1) * 4);
        tiles_.append((encode - 1) * 4 + 1);
        tiles_.append((encode - 1) * 4 + 2);
        tiles_.append((encode - 1) * 4 + 3);
      }
      else if (11u <= encode && encode <= 14u || 16u <= encode && encode <= 19u) {
        tiles_.append((encode - 2) * 4);
        tiles_.append((encode - 2) * 4 + 1);
        tiles_.append((encode - 2) * 4 + 2);
        tiles_.append((encode - 2) * 4 + 3);
      }
      else if (21u <= encode && encode <= 24u || 26u <= encode && encode < 37u) {
        tiles_.append((encode - 3) * 4);
        tiles_.append((encode - 3) * 4 + 1);
        tiles_.append((encode - 3) * 4 + 2);
        tiles_.append((encode - 3) * 4 + 3);
      }
      python::tuple t = python::make_tuple(python::str("kan"), tiles_, 1);
      fulu_list.append(t);
    }
  }

  return fulu_list;
}

void Shoupai::updateHupaiList_()
{
  KANACHAN_ASSERT((tingpai_cache_));
  KANACHAN_ASSERT((xiangting_lower_bound_ == 1u));

  hupai_list_.clear();

  std::vector<std::uint_fast8_t> shoupai34 = getShoupai34_();
  std::uint_fast8_t const num_fulu = getNumFulu_();
  for (std::uint_fast8_t i = 0u; i < 37u; ++i) {
    std::uint_fast8_t const hupai34 = [i]() -> std::uint_fast8_t {
      if (i == 0u) {
        return 4u;
      }
      if (/*1u <= i && */i <= 9u) {
        return i - 1u;
      }
      if (i == 10u) {
        return 13u;
      }
      if (/*11u <= i && */i <= 19u) {
        return i - 2u;
      }
      if (i == 20u) {
        return 22u;
      }
      if (/*21u <= i && */i < 37u) {
        return i - 3u;
      }
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }();
    if (shoupai34[hupai34] == 4u) {
      shoupai34[hupai34] -= 2u;
      std::uint_fast8_t const xiangting
        = Kanachan::calculateXiangting(shoupai34, 4u - num_fulu - 1u);
      if (xiangting == 0u) {
        hupai_list_.push_back(i);
      }
      shoupai34[hupai34] += 2u;
    }
    else {
      ++shoupai34[hupai34];
      std::uint_fast8_t const xiangting = Kanachan::calculateXiangting(shoupai34, 4u - num_fulu);
      if (xiangting == 0u) {
        hupai_list_.push_back(i);
      }
      --shoupai34[hupai34];
    }
  }

  KANACHAN_ASSERT((hupai_list_.size() >= 1u));
}

void Shoupai::appendToFeatures(std::vector<std::uint_fast16_t> &sparse_features) const
{
  for (std::uint_fast8_t i = 0u; i < 37u; ++i) {
    std::uint_fast8_t const num_tiles = shoupai_[i];
    for (std::uint_fast8_t j = 0u; j < num_tiles; ++j) {
      std::uint_fast16_t const feature = 337u + offsets_to_136_[i] + j;
      sparse_features.push_back(feature);
    }
  }
}

std::vector<std::uint_fast16_t> Shoupai::getCandidatesOnZimo(
  std::uint_fast8_t const zimo_tile, bool const first_zimo,
  bool const lizhi_prohibited, bool const gang_prohibited,
  long const tool_config) const
{
  KANACHAN_ASSERT((kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((zimo_tile < 37u));
  KANACHAN_ASSERT((tool_config >= 0));

  std::vector<std::uint_fast16_t> candidates;

  std::vector<std::uint_fast8_t> shoupai34 = getShoupai34_();
  std::uint_fast8_t const zimo_tile_34 = [zimo_tile]() -> std::uint_fast8_t {
    if (zimo_tile == 0u) {
      return 4u;
    }
    if (/*1u <= zimo_tile && */zimo_tile <= 9u) {
      return zimo_tile - 1u;
    }
    if (zimo_tile == 10u) {
      return 13u;
    }
    if (/*11u <= zimo_tile && */zimo_tile <= 19u) {
      return zimo_tile - 2u;
    }
    if (zimo_tile == 20u) {
      return 22u;
    }
    if (/*21u <= zimo_tile && */zimo_tile < 37u) {
      return zimo_tile - 3u;
    }
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }();
  shoupai34[zimo_tile_34] += 1u;

  // Check for Da Pai (打牌)
  if ((tool_config & (1 << 1)) != 0 || (tool_config & (1 << 7)) != 0) {
    // 立直中なので打牌に関する選択肢は基本的に存在しない．
    // ただし，暗槓の選択肢がある場合は自摸切りの選択肢が存在する．
    // (コード下部を参照のこと)
  }
  else {
    // 立直中ではない．
    std::uint_fast8_t const num_fulu = getNumFulu_();
    bool const menqian = isMenqian();
    bool const m0 = (shoupai_[0u] == 1u || zimo_tile == 0u);
    bool const p0 = (shoupai_[10u] == 1u || zimo_tile == 10u);
    bool const s0 = (shoupai_[20u] == 1u || zimo_tile == 20u);
    for (std::uint_fast8_t i = 0u; i < 34u; ++i) {
      if (shoupai34[i] == 0u) {
        continue;
      }
      --shoupai34[i];
      std::uint_fast8_t const xiangting = Kanachan::calculateXiangting(shoupai34, 4u - num_fulu);
      ++shoupai34[i];
      bool const lizhi = menqian && (xiangting == 1u) && !lizhi_prohibited;

      auto append = [&](std::uint_fast8_t const tile, bool const moqi) mutable -> void
      {
        KANACHAN_ASSERT((tile < 37u));

        if (moqi) {
          candidates.push_back(tile * 4u + 1u * 2u + 0u);
          if (lizhi) {
            candidates.push_back(tile * 4u + 1u * 2u + 1u);
          }
        }
        else {
          candidates.push_back(tile * 4u + 0u * 2u + 0u);
          if (lizhi) {
            candidates.push_back(tile * 4u + 0u * 2u + 1u);
          }
        }
      };

      // 親の配牌14枚には手牌と自摸牌の区別が無いので，親の第一打牌は常に
      // 手出しとなる．
      bool const zhuangjia_first_dapai = ((tool_config & (1u << 8u)) != 0u);
      if (i == 4u) {
        if (!m0) {
          if (zimo_tile == 5u) {
            append(5u, !zhuangjia_first_dapai);
            if (shoupai34[i] >= 2u && !zhuangjia_first_dapai) {
              append(5u, false);
            }
          }
          else {
            append(5u, false);
          }
        }
        else if (zimo_tile == 0u) {
          append(0u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 2u) {
            append(5u, false);
          }
        }
        else if (zimo_tile == 5u) {
          append(0u, false);
          append(5u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 3u && !zhuangjia_first_dapai) {
            append(5u, false);
          }
        }
        else {
          append(0u, false);
          if (shoupai34[i] >= 2u) {
            append(5u, false);
          }
        }
      }
      else if (i == 13u) {
        if (!p0) {
          if (zimo_tile == 15u) {
            append(15u, !zhuangjia_first_dapai);
            if (shoupai34[i] >= 2u && !zhuangjia_first_dapai) {
              append(15u, false);
            }
          }
          else {
            append(15u, false);
          }
        }
        else if (zimo_tile == 10u) {
          append(10u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 2u) {
            append(15u, false);
          }
        }
        else if (zimo_tile == 15u) {
          append(10u, false);
          append(15u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 3u && !zhuangjia_first_dapai) {
            append(15u, false);
          }
        }
        else {
          append(10u, false);
          if (shoupai34[i] >= 2u) {
            append(15u, false);
          }
        }
      }
      else if (i == 22u) {
        if (!s0) {
          if (zimo_tile == 25u) {
            append(25u, !zhuangjia_first_dapai);
            if (shoupai34[i] >= 2u && !zhuangjia_first_dapai) {
              append(25u, false);
            }
          }
          else {
            append(25u, false);
          }
        }
        else if (zimo_tile == 20u) {
          append(20u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 2u) {
            append(25u, false);
          }
        }
        else if (zimo_tile == 25u) {
          append(20u, false);
          append(25u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 3u && !zhuangjia_first_dapai) {
            append(25u, false);
          }
        }
        else {
          append(20u, false);
          if (shoupai34[i] >= 2u) {
            append(25u, false);
          }
        }
      }
      else if (i <= 3u || 5u <= i && i <= 8u) {
        if (zimo_tile == i + 1u) {
          append(i + 1u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 2u && !zhuangjia_first_dapai) {
            append(i + 1u, false);
          }
        }
        else {
          append(i + 1u, false);
        }
      }
      else if (9u <= i && i<= 12u || 14u <= i && i <= 17u) {
        if (zimo_tile == i + 2u) {
          append(i + 2u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 2u && !zhuangjia_first_dapai) {
            append(i + 2u, false);
          }
        }
        else {
          append(i + 2u, false);
        }
      }
      else if (18u <= i && i <= 21u || 23u <= i && i < 34u) {
        if (zimo_tile == i + 3u) {
          append(i + 3u, !zhuangjia_first_dapai);
          if (shoupai34[i] >= 2u && !zhuangjia_first_dapai) {
            append(i + 3u, false);
          }
        }
        else {
          append(i + 3u, false);
        }
      }
    }
  }

  if (!gang_prohibited && ((tool_config & (1 << 1)) != 0 || (tool_config & (1 << 7)) != 0)) {
    // 立直中の暗槓．送り槓を禁止するよう注意する．
    KANACHAN_ASSERT((hupai_list_.size() >= 1u));
    while (shoupai34[zimo_tile_34] == 4u) {
      std::vector<std::uint_fast8_t> new_shoupai34 = getShoupai34_();
      std::uint_fast8_t const num_fulu = getNumFulu_();
      new_shoupai34[zimo_tile_34] -= 3u;
      std::uint_fast8_t const xiangting
        = Kanachan::calculateXiangting(new_shoupai34, 4u - num_fulu - 1u);
      if (xiangting >= 2u) {
        // 暗槓すると聴牌が外れるので送り槓になる．
        break;
      }

      std::vector<std::uint_fast8_t> new_hupai_list;
      for (std::uint_fast8_t i = 0u; i < 37u; ++i) {
        std::uint_fast8_t const hupai34 = [i]() -> std::uint_fast8_t {
          if (i == 0u) {
            return 4u;
          }
          if (/*1u <= i && */i <= 9u) {
            return i - 1u;
          }
          if (i == 10u) {
            return 13u;
          }
          if (/*11u <= i && */i <= 19u) {
            return i - 2u;
          }
          if (i == 20u) {
            return 22u;
          }
          if (/*21u <= i && */i < 37u) {
            return i - 3u;
          }
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }();
        if (new_shoupai34[hupai34] == 4u) {
          new_shoupai34[hupai34] -= 2u;
          std::uint_fast8_t const xiangting
            = Kanachan::calculateXiangting(new_shoupai34, 4u - num_fulu - 2u);
          if (xiangting == 0u) {
            new_hupai_list.push_back(i);
          }
          new_shoupai34[hupai34] += 2u;
        }
        else {
          ++new_shoupai34[hupai34];
          std::uint_fast8_t const xiangting
            = Kanachan::calculateXiangting(new_shoupai34, 4u - num_fulu - 1u);
          if (xiangting == 0u) {
            new_hupai_list.push_back(i);
          }
          --new_shoupai34[hupai34];
        }
      }
      if (new_hupai_list != hupai_list_) {
        // 暗槓すると待ちが変わるので送り槓になる．
        break;
      }

      std::uint_fast16_t const encode = 148u + zimo_tile_34;
      candidates.push_back(encode);
      break;
    }
  }
  else if (!gang_prohibited) {
    // Check for An Gang (暗槓) without Li Zhi (立直)
    for (std::uint_fast8_t i = 0u; i < 34u; ++i) {
      KANACHAN_ASSERT((shoupai34[i] <= 4u));
      if (shoupai34[i] == 4u) {
        std::uint_fast16_t const encode = 148u + i;
        candidates.push_back(encode);
      }
    }

    // Check for Jia Gang (加槓)
    constexpr std::array<std::uint_fast8_t, 40u> jiagang_decoder{
      1u,
      2u,
      3u,
      4u,
      0u, // (5m, 5m, 5m)
      5u, // (0m, 5m, 5m)
      5u, // (5m, 5m, 0m)
      6u,
      7u,
      8u,
      9u,
      11u,
      12u,
      13u,
      14u,
      10u, // (5p, 5p, 5p)
      15u, // (0p, 5p, 5p)
      15u, // (5p, 5p, 0p)
      16u,
      17u,
      18u,
      19u,
      21u,
      22u,
      23u,
      24u,
      20u, // (5s, 5s, 5s)
      25u, // (0s, 5s, 5s)
      25u, // (5s, 5s, 0s)
      26u,
      27u,
      28u,
      29u,
      30u,
      31u,
      32u,
      33u,
      34u,
      35u,
      36u
    };
    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      std::uint_fast16_t const fulu = fulu_list_[i];
      if (fulu == std::numeric_limits<std::uint_fast16_t>::max()) {
        break;
      }
      KANACHAN_ASSERT((fulu < 392u));
      if (fulu < 90u || 210u <= fulu) {
        continue;
      }
      std::uint_fast8_t const relseat = (fulu - 90u) / 40u;
      std::uint_fast8_t const peng = (fulu - 90u - relseat * 40u);
      std::uint_fast8_t const tile_to_meld = jiagang_decoder[peng];
      if (shoupai_[tile_to_meld] == 1u || zimo_tile == tile_to_meld) {
        candidates.push_back(182u + tile_to_meld);
      }
    }
  }

  // Check for Zi Mo Hu (自摸和)
  if (std::find(hupai_list_.cbegin(), hupai_list_.cend(), zimo_tile) != hupai_list_.cend()) {
    Kanachan::GIL::RecursiveLock gil_lock;

    python::list fulu_list = getFuluList_();
    python::list shoupai136 = getShoupai136_(fulu_list, zimo_tile);
    python::list candidates_tmp;
    for (std::uint_fast16_t const candidate : candidates) {
      candidates_tmp.append(candidate);
    }
    external_tool_.attr("append_zimohu_candidate")(
      shoupai136, fulu_list, zimo_tile, tool_config, candidates_tmp);
    candidates.clear();
    for (python::ssize_t i = 0u; i < python::len(candidates_tmp); ++i) {
      std::uint_fast16_t const candidate = python::extract<long>(candidates_tmp[i])();
      candidates.push_back(candidate);
    }
    zhenting_ = true;
  }

  if (first_zimo) {
    // Check for Jiu Zhong Jiu Pai (九種九牌)
    std::uint_fast8_t num_kinds = 0u;
    if (shoupai34[0u] >= 1u) {
      ++num_kinds;
    }
    if (shoupai34[8u] >= 1u) {
      ++num_kinds;
    }
    if (shoupai34[9u] >= 1u) {
      ++num_kinds;
    }
    if (shoupai34[17u] >= 1u) {
      ++num_kinds;
    }
    if (shoupai34[18u] >= 1u) {
      ++num_kinds;
    }
    if (shoupai34[26u] >= 1u) {
      ++num_kinds;
    }
    for (std::uint_fast8_t i = 27u; i < 34u; ++i) {
      if (shoupai34[i] >= 1u) {
        ++num_kinds;
      }
    }
    if (num_kinds >= 9u) {
      candidates.push_back(220u);
    }
  }

  if ((tool_config & (1 << 1)) != 0 || (tool_config & (1 << 7)) != 0) {
    // 立直中．
    if (candidates.size() >= 1u) {
      // 立直中に選択肢がある場合は自摸切りの選択肢を加える．
      candidates.push_back(zimo_tile * 4u + 1u * 2u + 0u);
    }
  }

  std::sort(candidates.begin(), candidates.end());
  return candidates;
}

std::vector<std::uint_fast16_t> Shoupai::getCandidatesOnDapai(
  std::uint_fast8_t const relseat, std::uint_fast8_t const dapai,
  bool const gang_prohibited, long const tool_config) const
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((relseat < 3u));
  KANACHAN_ASSERT((dapai < 37u));
  KANACHAN_ASSERT((tool_config >= 0));

  std::vector<std::uint_fast16_t> candidates;

  if ((tool_config & (1 << 1)) == 0 && (tool_config & (1 << 6)) == 0 && (tool_config & (1 << 7)) == 0) {
    // 立直中でない場合かつ河底牌以外の場合 (立直中または河底牌は鳴けない)

    if (relseat == 2u) {
      // Check for Chi (チー)
      std::uint_fast8_t const num_shoupai = std::accumulate(
        shoupai_.cbegin(), shoupai_.cend(), static_cast<std::uint_fast8_t>(0u));

      for (std::uint_fast8_t i = 0u; i < chi_encode_.size(); ++i) {
        auto const [t0, t1, d, kuikae_] = chi_encode_[i];
        if (d != dapai) {
          continue;
        }
        {
          // 仮にチーした場合に喰い替えによって全ての打牌が禁止されるならば，
          // そもそもそのようなチーができない．
          std::uint_fast8_t const num_tiles
            = std::accumulate(shoupai_.cbegin(), shoupai_.cend(), 0u);
          std::uint_fast8_t count = 0u;
          count += std::min<std::uint_fast8_t>(1u, shoupai_[t0]);
          count += std::min<std::uint_fast8_t>(1u, shoupai_[t1]);
          count += shoupai_[d];
          if (kuikae_ != std::numeric_limits<std::uint_fast8_t>::max()) {
            count += shoupai_[kuikae_];
          }
          if (d == 0u || kuikae_ == 0u) {
            count += shoupai_[5u];
          }
          else if (d == 5u || kuikae_ == 5u) {
            count += shoupai_[0u];
          }
          else if (d == 10u || kuikae_ == 10u) {
            count += shoupai_[15u];
          }
          else if (d == 15u || kuikae_ == 15u) {
            count += shoupai_[10u];
          }
          else if (d == 20u || kuikae_ == 20u) {
            count += shoupai_[25u];
          }
          else if (d == 25u || kuikae_ == 25u) {
            count += shoupai_[20u];
          }
          KANACHAN_ASSERT((count <= num_tiles));
          if (count == num_tiles) {
            continue;
          }
        }
        if (shoupai_[t0] >= 1u && shoupai_[t1] >= 1u) {
          candidates.push_back(222u + i);
        }
      }
    }

    // Check for Peng (ポン)
    for (std::uint_fast8_t i = 0u; i < peng_encode_.size(); ++i) {
      auto const [t0, t1, d] = peng_encode_[i];
      if (d != dapai) {
        continue;
      }
      if (t0 == t1 && shoupai_[t0] >= 2u) {
        candidates.push_back(312u + relseat * 40u + i);
      }
      if (t0 != t1 && shoupai_[t0] >= 1u && shoupai_[t1] >= 1u) {
        candidates.push_back(312u + relseat * 40u + i);
      }
    }

    if (!gang_prohibited) {
      // Check for Da Ming Gang (大明槓)
      for (std::uint_fast8_t i = 0u; i <= 20u; i += 10u) {
        if (dapai == i + 0u && shoupai_[i + 5u] == 3u) {
          candidates.push_back(432u + relseat * 37u + i);
        }
        for (std::uint_fast8_t j = i + 1u; j <= i + 4u; ++j) {
          if (dapai == j && shoupai_[j] == 3u) {
            candidates.push_back(432u + relseat * 37u + j);
          }
        }
        if (dapai == i + 5u && shoupai_[i + 0u] == 1u && shoupai_[i + 5u] == 2u) {
          candidates.push_back(432u + relseat * 37u + i + 5u);
        }
        for (std::uint_fast8_t j = i + 6u; j <= i + 9u; ++j) {
          if (dapai == j && shoupai_[j] == 3u) {
            candidates.push_back(432u + relseat * 37u + j);
          }
        }
      }
      for (std::uint_fast8_t i = 30u; i < 37u; ++i) {
        if (dapai == i && shoupai_[i] == 3u) {
          candidates.push_back(432u + relseat * 37u + i);
        }
      }
    }
  }

  // Check for Rong (栄和)
  if (!zhenting_ && std::find(hupai_list_.cbegin(), hupai_list_.cend(), dapai) != hupai_list_.cend()){
    Kanachan::GIL::RecursiveLock gil_lock;

    python::list fulu_list = getFuluList_();
    python::list shoupai136 = getShoupai136_(fulu_list, dapai);
    python::list candidates_tmp;
    for (std::uint_fast16_t const candidate : candidates) {
      candidates_tmp.append(candidate);
    }
    external_tool_.attr("append_rong_candidate")(
      relseat, shoupai136, fulu_list, dapai, tool_config, candidates_tmp);
    candidates.clear();
    for (python::ssize_t i = 0u; i < python::len(candidates_tmp); ++i) {
      std::uint_fast16_t const candidate = python::extract<long>(candidates_tmp[i])();
      candidates.push_back(candidate);
    }
    zhenting_ = true;
  }

  if (candidates.size() >= 1u) {
    // Skip
    candidates.push_back(221u);
  }

  std::sort(candidates.begin(), candidates.end());
  return candidates;
}

std::vector<std::uint_fast16_t> Shoupai::getCandidatesOnChiPeng() const
{
  std::array<std::uint_fast8_t, 3u> kuikae = {
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max()
  };
  if (kuikae_delayed_ < 90u) {
    // チーにおける喰い替えを禁止する．
    auto const [tile0_, tile1_, kuikae0, kuikae1] = chi_encode_[kuikae_delayed_];
    if (kuikae0 == 0u || kuikae0 == 5u) {
      kuikae = { 0u, 5u, kuikae1 };
    }
    else if (kuikae1 == 0u || kuikae1 == 5u) {
      kuikae = { 0u, 5u, kuikae0 };
    }
    else if (kuikae0 == 10u || kuikae0 == 15u) {
      kuikae = { 10u, 15u, kuikae1 };
    }
    else if (kuikae1 == 10u || kuikae1 == 15u) {
      kuikae = { 10u, 15u, kuikae0 };
    }
    else if (kuikae0 == 20u || kuikae0 == 25u) {
      kuikae = { 20u, 25u, kuikae1 };
    }
    else if (kuikae1 == 20u || kuikae1 == 25u) {
      kuikae = { 20u, 25u, kuikae0 };
    }
    else {
      kuikae = { kuikae0, kuikae1, std::numeric_limits<std::uint_fast8_t>::max() };
    }
  }
  else {
    // ポンにおける喰い替えを禁止する．
    KANACHAN_ASSERT((/*90u <= kuikae_delayed_ && */kuikae_delayed_ < 210u));
    std::uint_fast8_t const relseat = (kuikae_delayed_ - 90u) / 40u;
    std::uint_fast8_t const peng = (kuikae_delayed_ - 90u - relseat * 40u);
    auto const [tile0_, tile1_, kuikae_] = peng_encode_[peng];
    if (kuikae_ == 0u) {
      kuikae = { 0u, 5u, std::numeric_limits<std::uint_fast8_t>::max() };
    }
    else if (kuikae_ == 5u) {
      kuikae = { 0u, 5u, std::numeric_limits<std::uint_fast8_t>::max() };
    }
    else if (kuikae_ == 10u) {
      kuikae = { 10u, 15u, std::numeric_limits<std::uint_fast8_t>::max() };
    }
    else if (kuikae_ == 15u) {
      kuikae = { 10u, 15u, std::numeric_limits<std::uint_fast8_t>::max() };
    }
    else if (kuikae_ == 20u) {
      kuikae = { 20u, 25u, std::numeric_limits<std::uint_fast8_t>::max() };
    }
    else if (kuikae_ == 25u) {
      kuikae = { 20u, 25u, std::numeric_limits<std::uint_fast8_t>::max() };
    }
    else {
      kuikae = {
        kuikae_,
        std::numeric_limits<std::uint_fast8_t>::max(),
        std::numeric_limits<std::uint_fast8_t>::max()
      };
    }
  }

  std::vector<std::uint_fast16_t> candidates;
  for (std::uint_fast8_t i = 0u; i < shoupai_.size(); ++i) {
    if (std::find(kuikae.cbegin(), kuikae.cend(), i) != kuikae.cend()) {
      continue;
    }
    if (shoupai_[i] >= 1u) {
      std::uint_fast16_t const encode = i * 4u + 0 * 2u + 0u;
      candidates.push_back(encode);
    }
  }

  return candidates;
}

std::vector<std::uint_fast16_t> Shoupai::getCandidatesOnAngang(
  std::uint_fast8_t const relseat, std::uint_fast8_t const encode) const
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((relseat < 3u));
  KANACHAN_ASSERT((encode < 34u));

  if (!isMenqian()) {
    return std::vector<std::uint_fast16_t>();
  }

  if (zhenting_) {
    return std::vector<std::uint_fast16_t>();
  }

  std::uint_fast8_t n = 0u;
  bool head = false;
  if (shoupai_[1u] >= 1u || encode == 0u) {
    // 1m
    ++n;
    if (shoupai_[1u] == 2u || shoupai_[1u] == 1u && encode == 0u) {
      head = true;
    }
  }
  else {
    return std::vector<std::uint_fast16_t>();
  }
  if (shoupai_[9u] >= 1u || encode == 8u) {
    // 9m
    ++n;
    if (shoupai_[9u] == 2u || shoupai_[9u] == 1u && encode == 8u) {
      head = true;
    }
  }
  else {
    return std::vector<std::uint_fast16_t>();
  }
  if (shoupai_[11u] >= 1u || encode == 9u) {
    // 1p
    ++n;
    if (shoupai_[11u] == 2u || shoupai_[11u] == 1u && encode == 9u) {
      head = true;
    }
  }
  else {
    return std::vector<std::uint_fast16_t>();
  }
  if (shoupai_[19u] >= 1u || encode == 17u) {
    // 9p
    ++n;
    if (shoupai_[19u] == 2u || shoupai_[19u] == 1u && encode == 17u) {
      head = true;
    }
  }
  else {
    return std::vector<std::uint_fast16_t>();
  }
  if (shoupai_[21u] >= 1u || encode == 18u) {
    // 1s
    ++n;
    if (shoupai_[21u] == 2u || shoupai_[21u] == 1u && encode == 18u) {
      head = true;
    }
  }
  else {
    return std::vector<std::uint_fast16_t>();
  }
  if (shoupai_[29u] >= 1u || encode == 26u) {
    // 9s
    ++n;
    if (shoupai_[29u] == 2u || shoupai_[29u] == 1u && encode == 26u) {
      head = true;
    }
  }
  else {
    return std::vector<std::uint_fast16_t>();
  }
  for (std::uint_fast8_t i = 27u; i < 34u; ++i) {
    if (shoupai_[i + 3u] >= 1u || encode == i) {
      ++n;
      if (shoupai_[i + 3u] == 2u || shoupai_[i + 3u] == 1u && encode == i) {
        head = true;
      }
    }
    else {
      return std::vector<std::uint_fast16_t>();
    }
  }

  if (n < 13u || !head) {
    return std::vector<std::uint_fast16_t>();
  }
  KANACHAN_ASSERT((n == 14u && head));

  std::vector<std::uint_fast16_t> candidate;
  candidate.push_back(221u);
  candidate.push_back(543u + relseat);
  // 槍槓を見逃した場合のフリテン
  zhenting_ = true;
  return candidate;
}

std::vector<std::uint_fast16_t> Shoupai::getCandidatesOnJiagang(
  std::uint_fast8_t const relseat, std::uint_fast8_t const encode,
  long const tool_config) const
{
  KANACHAN_ASSERT((kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((relseat < 3u));
  KANACHAN_ASSERT((encode < 37u));

  if (zhenting_) {
    return std::vector<std::uint_fast16_t>();
  }

  std::vector<std::uint_fast16_t> candidates;

  // Check for Qiang Gang (槍槓)
  {
    Kanachan::GIL::RecursiveLock gil_lock;

    python::list fulu_list = getFuluList_();
    python::list shoupai136 = getShoupai136_(fulu_list, encode);
    python::list candidates_tmp;
    external_tool_.attr("append_rong_candidate")(
      relseat, shoupai136, fulu_list, encode, tool_config, candidates_tmp);
    for (python::ssize_t i = 0u; i < python::len(candidates_tmp); ++i) {
      std::uint_fast16_t const candidate = python::extract<long>(candidates_tmp[i])();
      candidates.push_back(candidate);
    }
  }

  if (candidates.size() >= 1u) {
    // Skip
    candidates.push_back(221u);
    // 槍槓を見逃した場合のフリテン
    zhenting_ = true;
  }

  std::sort(candidates.begin(), candidates.end());
  return candidates;
}

std::pair<std::uint_fast8_t, std::uint_fast8_t> Shoupai::calculateHand(
  std::uint_fast8_t const hupai, std::vector<std::uint_fast8_t> const &dora_indicators,
  long const tool_config) const
{
  KANACHAN_ASSERT((kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((hupai < 37u));
  KANACHAN_ASSERT((tool_config >= 0));

  Kanachan::GIL::RecursiveLock gil_lock;

  python::list fulu_list = getFuluList_();
  python::list shoupai136 = getShoupai136_(fulu_list, hupai);
  python::list dora_indicators_tmp;
  for (std::uint_fast8_t const dora_indicator : dora_indicators) {
    dora_indicators_tmp.append(dora_indicator);
  }
  python::object o = external_tool_.attr("calculate_hand")(
    shoupai136, fulu_list, hupai, dora_indicators_tmp, tool_config);
  python::extract<python::tuple> e(o);
  if (!e.check()) {
    KANACHAN_THROW<std::runtime_error>("A type error.");
  }
  python::tuple t = e();

  if (python::len(t) != 2u) {
    KANACHAN_THROW<std::runtime_error>("A length error.");
  }
  python::object fan = t[0];
  python::object fu = t[1];

  python::extract<long> fan_(fan);
  if (!fan_.check()) {
    KANACHAN_THROW<std::runtime_error>("A type error.");
  }
  python::extract<long> fu_(fu);
  if (!fu_.check()) {
    KANACHAN_THROW<std::runtime_error>("A type error.");
  }

  return std::pair<std::uint_fast8_t, std::uint_fast8_t>(fan_(), fu_());
}

void Shoupai::onPostZimo(
  std::uint_fast8_t const zimo_tile, std::uint_fast8_t const dapai,
  bool const in_lizhi)
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((zimo_tile < 37u));
  KANACHAN_ASSERT((dapai < 37u));

  he_.push_back(dapai);

  if (zimo_tile == dapai) {
    if (in_lizhi) {
      // 立直中なのでフリテンに関する状態は変わらない．
      return;
    }
    // 立直中ではないのでフリテンに関する状態を更新する．
    zhenting_ = false;
    for (std::uint_fast8_t const hupai : hupai_list_) {
      auto const found = std::find(he_.cbegin(), he_.cend(), hupai);
      if (found != he_.cend()) {
        zhenting_ = true;
      }
    }
    return;
  }

  // これ以降は手替わりが発生した場合の処理．
  KANACHAN_ASSERT((!in_lizhi));
  KANACHAN_ASSERT((shoupai_[dapai] >= 1u));

  ++shoupai_[zimo_tile];
  --shoupai_[dapai];

  tingpai_cache_ = false;
  if (xiangting_lower_bound_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    if (xiangting_lower_bound_ >= 1u) {
      --xiangting_lower_bound_;
    }
  }
  hupai_list_.clear();
  zhenting_ = false;

  if (isTingpai()) {
    updateHupaiList_();
  }
  else {
    hupai_list_.clear();
  }

  for (std::uint_fast8_t const hupai : hupai_list_) {
    auto const found = std::find(he_.cbegin(), he_.cend(), hupai);
    if (found != he_.cend()) {
      zhenting_ = true;
    }
  }
}

void Shoupai::onChi(std::uint_fast8_t const encode)
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((encode < 90u));

  auto const [tile0, tile1, dapai, kuikae] = chi_encode_[encode];
  if (shoupai_[tile0] == 0u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << encode << ": An invalid Chi.";
  }
  --shoupai_[tile0];
  if (shoupai_[tile1] == 0u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << encode << ": An invalid Chi.";
  }
  --shoupai_[tile1];

  std::uint_fast8_t i = 0u;
  for (; i < 4u; ++i) {
    if (fulu_list_[i] == std::numeric_limits<std::uint_fast16_t>::max()) {
      break;
    }
  }
  if (i == 4u) {
    KANACHAN_THROW<std::logic_error>("No more Fu Lu.");
  }
  fulu_list_[i] = encode;

  kuikae_delayed_ = encode;
}

void Shoupai::onPeng(
  std::uint_fast8_t const relseat, std::uint_fast8_t const encode)
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((relseat < 3u));
  KANACHAN_ASSERT((encode < 90u));

  auto const [tile0, tile1, dapai] = peng_encode_[encode];
  if (shoupai_[tile0] == 0u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << encode << ": An invalid Peng.";
  }
  --shoupai_[tile0];
  if (shoupai_[tile1] == 0u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << encode << ": An invalid Peng.";
  }
  --shoupai_[tile1];

  std::uint_fast8_t i = 0u;
  for (; i < 4u; ++i) {
    if (fulu_list_[i] == std::numeric_limits<std::uint_fast16_t>::max()) {
      break;
    }
  }
  if (i == 4u) {
    KANACHAN_THROW<std::logic_error>("No more Fu Lu.");
  }
  fulu_list_[i] = 90u + relseat * 40u + encode;

  kuikae_delayed_ = 90u + encode;
}

void Shoupai::onPostChiPeng(std::uint_fast8_t const dapai)
{
  KANACHAN_ASSERT((kuikae_delayed_ < 210u));
  KANACHAN_ASSERT((dapai < 37u));
  KANACHAN_ASSERT((shoupai_[dapai] >= 1u));

  if (kuikae_delayed_ < 90u) {
    // チーに対する喰い替えの禁止を確認する．
    auto const [tile0_, tile1_, kuikae0, kuikae1] = chi_encode_[kuikae_delayed_];
    if (kuikae0 == 0u) {
      KANACHAN_ASSERT((dapai != 0u));
      KANACHAN_ASSERT((dapai != 5u));
    }
    else if (kuikae0 == 10u) {
      KANACHAN_ASSERT((dapai != 10u));
      KANACHAN_ASSERT((dapai != 15u));
    }
    else if (kuikae0 == 20u) {
      KANACHAN_ASSERT((dapai != 20u));
      KANACHAN_ASSERT((dapai != 25u));
    }
    else {
      KANACHAN_ASSERT((dapai != kuikae0));
    }
    if (kuikae1 != std::numeric_limits<std::uint_fast16_t>::max()) {
      if (kuikae1 == 0u) {
        KANACHAN_ASSERT((dapai != 0u));
        KANACHAN_ASSERT((dapai != 5u));
      }
      else if (kuikae1 == 10u) {
        KANACHAN_ASSERT((dapai != 10u));
        KANACHAN_ASSERT((dapai != 15u));
      }
      else if (kuikae1 == 20u) {
        KANACHAN_ASSERT((dapai != 20u));
        KANACHAN_ASSERT((dapai != 25u));
      }
      else {
        KANACHAN_ASSERT((dapai != kuikae1));
      }
    }
  }
  else if (/*90u <= kuikae_delayed_ && */kuikae_delayed_ < 210u) {
    // ポンに対する喰い替えの禁止を確認する．
    std::uint_fast8_t const relseat = (kuikae_delayed_ - 90u) / 40u;
    std::uint_fast8_t const peng = (kuikae_delayed_ - 90u - relseat * 40u);
    auto const [tile0_, tile1_, kuikae] = peng_encode_[peng];
    if (kuikae == 0u) {
      KANACHAN_ASSERT((dapai != 0u));
      KANACHAN_ASSERT((dapai != 5u));
    }
    else if (kuikae == 10u) {
      KANACHAN_ASSERT((dapai != 10u));
      KANACHAN_ASSERT((dapai != 15u));
    }
    else if (kuikae == 20u) {
      KANACHAN_ASSERT((dapai != 20u));
      KANACHAN_ASSERT((dapai != 25u));
    }
    else {
      KANACHAN_ASSERT((dapai != kuikae));
    }
  }

  kuikae_delayed_ = std::numeric_limits<std::uint_fast16_t>::max();

  --shoupai_[dapai];
  he_.push_back(dapai);

  tingpai_cache_ = false;
  if (xiangting_lower_bound_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    if (xiangting_lower_bound_ >= 1u) {
      --xiangting_lower_bound_;
    }
  }
  hupai_list_.clear();
  zhenting_ = false;

  if (isTingpai()) {
    updateHupaiList_();
  }
  else {
    hupai_list_.clear();
  }

  for (std::uint_fast8_t const hupai : hupai_list_) {
    auto const found = std::find(he_.cbegin(), he_.cend(), hupai);
    if (found != he_.cend()) {
      zhenting_ = true;
    }
  }
}

void Shoupai::onDaminggang(
  std::uint_fast8_t const relseat, std::uint_fast8_t const dapai)
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((relseat < 3u));
  KANACHAN_ASSERT((dapai < 37u));

  auto const [tile0, tile1, tile2] = daminggang_encode_[dapai];
  if (shoupai_[tile0] == 0u) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << dapai << ": An invalid Da Ming Gang.";
  }
  --shoupai_[tile0];
  if (shoupai_[tile1] == 0u) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << dapai << ": An invalid Da Ming Gang.";
  }
  --shoupai_[tile1];
  if (shoupai_[tile2] == 0u) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << dapai << ": An invalid Da Ming Gang.";
  }
  --shoupai_[tile2];

  std::uint_fast8_t i = 0u;
  for (; i < 4u; ++i) {
    if (fulu_list_[i] == std::numeric_limits<std::uint_fast16_t>::max()) {
      break;
    }
  }
  if (i == 4u) {
    KANACHAN_THROW<std::logic_error>("No more Fu Lu.");
  }
  fulu_list_[i] = 210u + relseat * 37u + dapai;
}

void Shoupai::onAngang(
  std::uint_fast8_t zimo_tile, std::uint_fast8_t const encode)
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((zimo_tile < 37u));
  KANACHAN_ASSERT((encode < 34u));

  constexpr std::array<std::array<std::uint_fast8_t, 4u>, 34u> angang_encode = {
    std::array<std::uint_fast8_t, 4u>{ 1u, 1u, 1u, 1u },
    std::array<std::uint_fast8_t, 4u>{ 2u, 2u, 2u, 2u },
    std::array<std::uint_fast8_t, 4u>{ 3u, 3u, 3u, 3u },
    std::array<std::uint_fast8_t, 4u>{ 4u, 4u, 4u, 4u },
    std::array<std::uint_fast8_t, 4u>{ 5u, 5u, 5u, 0u },
    std::array<std::uint_fast8_t, 4u>{ 6u, 6u, 6u, 6u },
    std::array<std::uint_fast8_t, 4u>{ 7u, 7u, 7u, 7u },
    std::array<std::uint_fast8_t, 4u>{ 8u, 8u, 8u, 8u },
    std::array<std::uint_fast8_t, 4u>{ 9u, 9u, 9u, 9u },
    std::array<std::uint_fast8_t, 4u>{ 11u, 11u, 11u, 11u },
    std::array<std::uint_fast8_t, 4u>{ 12u, 12u, 12u, 12u },
    std::array<std::uint_fast8_t, 4u>{ 13u, 13u, 13u, 13u },
    std::array<std::uint_fast8_t, 4u>{ 14u, 14u, 14u, 14u },
    std::array<std::uint_fast8_t, 4u>{ 15u, 15u, 15u, 10u },
    std::array<std::uint_fast8_t, 4u>{ 16u, 16u, 16u, 16u },
    std::array<std::uint_fast8_t, 4u>{ 17u, 17u, 17u, 17u },
    std::array<std::uint_fast8_t, 4u>{ 18u, 18u, 18u, 18u },
    std::array<std::uint_fast8_t, 4u>{ 19u, 19u, 19u, 19u },
    std::array<std::uint_fast8_t, 4u>{ 21u, 21u, 21u, 21u },
    std::array<std::uint_fast8_t, 4u>{ 22u, 22u, 22u, 22u },
    std::array<std::uint_fast8_t, 4u>{ 23u, 23u, 23u, 23u },
    std::array<std::uint_fast8_t, 4u>{ 24u, 24u, 24u, 24u },
    std::array<std::uint_fast8_t, 4u>{ 25u, 25u, 25u, 20u },
    std::array<std::uint_fast8_t, 4u>{ 26u, 26u, 26u, 26u },
    std::array<std::uint_fast8_t, 4u>{ 27u, 27u, 27u, 27u },
    std::array<std::uint_fast8_t, 4u>{ 28u, 28u, 28u, 28u },
    std::array<std::uint_fast8_t, 4u>{ 29u, 29u, 29u, 29u },
    std::array<std::uint_fast8_t, 4u>{ 30u, 30u, 30u, 30u },
    std::array<std::uint_fast8_t, 4u>{ 31u, 31u, 31u, 31u },
    std::array<std::uint_fast8_t, 4u>{ 32u, 32u, 32u, 32u },
    std::array<std::uint_fast8_t, 4u>{ 33u, 33u, 33u, 33u },
    std::array<std::uint_fast8_t, 4u>{ 34u, 34u, 34u, 34u },
    std::array<std::uint_fast8_t, 4u>{ 35u, 35u, 35u, 35u },
    std::array<std::uint_fast8_t, 4u>{ 36u, 36u, 36u, 36u }
  };
  std::array<std::uint_fast8_t, 4u> tiles = angang_encode[encode];

  {
    std::uint_fast8_t count = 0u;
    for (std::uint_fast8_t i = 0u; i < shoupai_.size(); ++i) {
      for (std::uint_fast8_t j = 0u; j < tiles.size(); ++j) {
        if (shoupai_[tiles[j]] >= 1u) {
          --shoupai_[tiles[j]];
          ++count;
        }
        else if (zimo_tile == tiles[j]) {
          zimo_tile = std::numeric_limits<std::uint_fast8_t>::max();
          ++count;
        }
      }
    }
    if (count != 4u) {
      KANACHAN_THROW<std::logic_error>(_1)
        << '(' << static_cast<unsigned>(zimo_tile) << ", "
        << static_cast<unsigned>(encode) << "): An invalid An Gang.";
    }
  }

  if (zimo_tile != std::numeric_limits<std::uint_fast8_t>::max()) {
    ++shoupai_[zimo_tile];
  }

  {
    std::uint_fast8_t i = 0u;
    for (; i < 4u; ++i) {
      if (fulu_list_[i] == std::numeric_limits<std::uint_fast16_t>::max()) {
        break;
      }
    }
    if (i == 4u) {
      KANACHAN_THROW<std::logic_error>("No more Fu Lu.");
    }
    fulu_list_[i] = 321 + encode;
  }
}

void Shoupai::onJiagang(
  std::uint_fast8_t const zimo_tile, std::uint_fast8_t const encode)
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));
  KANACHAN_ASSERT((zimo_tile < 37u));
  KANACHAN_ASSERT((encode < 37u));
  KANACHAN_ASSERT((zimo_tile == encode || shoupai_[encode] >= 1u));

  if (zimo_tile != encode) {
    ++shoupai_[zimo_tile];
    --shoupai_[encode];
  }

  {
    std::uint_fast8_t i = 0u;
    for (; i < fulu_list_.size(); ++i) {
      std::uint_fast16_t const fulu = fulu_list_[i];
      if (fulu < 90u || 210 <= fulu) {
        continue;
      }
      std::uint_fast8_t const relseat = (fulu - 90u) / 40u;
      std::uint_fast8_t const peng = (fulu - 90u - relseat * 40u);
      if (encode == 0u && peng == 4u) {
        break;
      }
      if (1u <= encode && encode <= 4u && peng == encode - 1u) {
        break;
      }
      if (encode == 5u && (peng == 5u || peng == 6u)) {
        break;
      }
      if (6u <= encode && encode <= 9u && peng == encode + 1u) {
        break;
      }
      if (encode == 10u && peng == 15u) {
        break;
      }
      if (11u <= encode && encode <= 14u && peng == encode) {
        break;
      }
      if (encode == 15u && (peng == 16u || peng == 17u)) {
        break;
      }
      if (16u <= encode && encode <= 19u && peng == encode + 2u) {
        break;
      }
      if (encode == 20u && peng == 26u) {
        break;
      }
      if (21u <= encode && encode <= 24u && peng == encode + 1u) {
        break;
      }
      if (encode == 25u && (peng == 27u || peng == 28u)) {
        break;
      }
      if (26u <= encode && encode < 37u && peng == encode + 3u) {
        break;
      }
    }
    KANACHAN_ASSERT((i < fulu_list_.size()));

    fulu_list_[i] = 355u + encode;
  }
}

void Shoupai::onPostGang(bool const in_lizhi)
{
  KANACHAN_ASSERT(
    (kuikae_delayed_ == std::numeric_limits<std::uint_fast16_t>::max()));

  tingpai_cache_ = false;
  if (xiangting_lower_bound_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    if (xiangting_lower_bound_ >= 1u) {
      --xiangting_lower_bound_;
    }
  }
  hupai_list_.clear();

  if (isTingpai()) {
    updateHupaiList_();
  }
  else {
    hupai_list_.clear();
  }

  if (in_lizhi) {
    // 立直中なのでフリテンに関する状態は変わらない．
    return;
  }

  // 立直中ではないのでフリテンに関する状態を更新する．
  zhenting_ = false;
  for (std::uint_fast8_t const hupai : hupai_list_) {
    auto const found = std::find(he_.cbegin(), he_.cend(), hupai);
    if (found != he_.cend()) {
      zhenting_ = true;
    }
  }
}

} // namespace Kanachan
