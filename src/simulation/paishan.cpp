#include "simulation/paishan.hpp"

#include "simulation/gil.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <random>
#include <algorithm>
#include <vector>
#include <functional>
#include <utility>
#include <stdexcept>
#include <cstddef>


namespace {

using std::placeholders::_1;
namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

void swap(Paishan &lhs, Paishan &rhs) noexcept
{
  lhs.swap(rhs);
}

Paishan::Paishan(std::vector<std::uint_least32_t> const &seed)
  : tiles_({
       0u,  1u,  1u,  1u,  1u,  2u,  2u,  2u,  2u,  3u,  3u,  3u,  3u,  4u,  4u,  4u,  4u,
       5u,  5u,  5u,  6u,  6u,  6u,  6u,  7u,  7u,  7u,  7u,  8u,  8u,  8u,  8u,  9u,  9u,  9u,  9u,
      10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u,
      15u, 15u, 15u, 16u, 16u, 16u, 16u, 17u, 17u, 17u, 17u, 18u, 18u, 18u, 18u, 19u, 19u, 19u, 19u,
      20u, 21u, 21u, 21u, 21u, 22u, 22u, 22u, 22u, 23u, 23u, 23u, 23u, 24u, 24u, 24u, 24u,
      25u, 25u, 25u, 26u, 26u, 26u, 26u, 27u, 27u, 27u, 27u, 28u, 28u, 28u, 28u, 29u, 29u, 29u, 29u,
      30u, 30u, 30u, 30u, 31u, 31u, 31u, 31u, 32u, 32u, 32u, 32u, 33u, 33u, 33u, 33u,
      34u, 34u, 34u, 34u, 35u, 35u, 35u, 35u, 36u, 36u, 36u, 36u,
    })
{
  std::seed_seq sseq(seed.cbegin(), seed.cend());
  std::mt19937 urng(sseq);
  std::shuffle(tiles_.begin(), tiles_.end(), urng);
}

Paishan::Paishan(python::list paishan)
  : tiles_()
{
  if (paishan.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`paishan` must not be a `None`.";
  }

  Kanachan::GIL::RecursiveLock gil_lock;

  python::ssize_t const length = python::len(paishan);
  if (length != 136u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "paishan: An wrong length (" << length << ").";
  }
  for (python::ssize_t i = 0; i < length; ++i) {
    long const tile = [&](){
      python::extract<long> tile(paishan[i]);
      if (!tile.check()) {
        KANACHAN_THROW<std::invalid_argument>(_1) << "paishan: A type error.";
      }
      return tile();
    }();
    if (tile < 0 || 37 <= tile) {
      KANACHAN_THROW<std::invalid_argument>(_1) << "paishan: An wrong tile (" << tile << ").";
    }
    tiles_[i] = tile;
  }
}

void Paishan::swap(Paishan &rhs) noexcept
{
  using std::swap;
  swap(tiles_, rhs.tiles_);
}

Paishan &Paishan::operator=(Paishan const &rhs)
{
  Paishan(rhs).swap(*this);
  return *this;
}

Paishan &Paishan::operator=(Paishan &&rhs) noexcept
{
  Paishan(std::move(rhs)).swap(*this);
  return *this;
}

std::uint_fast8_t Paishan::operator[](std::uint_fast8_t const index) const
{
  KANACHAN_ASSERT((index < 136u));
  return tiles_[index];
}

} // namespace Kanachan
