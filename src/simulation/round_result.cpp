#include "simulation/round_result.hpp"

#include "common/assert.hpp"
#include "common/throw.hpp"
#include <functional>
#include <cstdint>


namespace{

using std::placeholders::_1;

}

namespace Kanachan{

RoundResult::RoundResult()
  : type_(UINT_FAST8_MAX)
  , in_lizhi_(false)
  , has_fulu_(false)
  , round_delta_score_(INT_FAST32_MAX)
  , round_score_(INT_FAST32_MAX)
{}

void RoundResult::setType(std::uint_fast8_t const type)
{
  if (type > 14u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(type);
  }
  type_ = type;
}

void RoundResult::setInLizhi(bool const in_lizhi)
{
  in_lizhi_ = in_lizhi;
}

void RoundResult::setHasFulu(bool const has_fulu)
{
  has_fulu_ = has_fulu;
}

void RoundResult::setRoundDeltaScore(std::int_fast32_t const round_delta_score)
{
  round_delta_score_ = round_delta_score;
}

void RoundResult::setRoundScore(std::int_fast32_t const round_score)
{
  round_score_ = round_score;
}

std::uint_fast8_t RoundResult::getType() const noexcept
{
  KANACHAN_ASSERT((type_ <= 14u || type_ == UINT_FAST8_MAX));
  return type_;
}

bool RoundResult::getInLizhi() const noexcept
{
  return in_lizhi_;
}

bool RoundResult::getHasFulu() const noexcept
{
  return has_fulu_;
}

std::int_fast32_t RoundResult::getRoundDeltaScore() const noexcept
{
  if (round_delta_score_ == INT_FAST32_MAX) {
    KANACHAN_THROW<std::runtime_error>("An uninitialized object.");
  }
  return round_delta_score_;
}

std::int_fast32_t RoundResult::getRoundScore() const noexcept
{
  if (round_score_ == INT_FAST32_MAX) {
    KANACHAN_THROW<std::runtime_error>("An uninitialized object.");
  }
  return round_score_;
}

}
