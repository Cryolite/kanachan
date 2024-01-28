#include "simulation/game_state.hpp"

#include "simulation/decision_maker.hpp"
#include "simulation/game_log.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <stop_token>
#include <array>
#include <functional>
#include <stdexcept>
#include <cstdint>


namespace {

using std::placeholders::_1;

} // namespace `anonymous

namespace Kanachan{

GameState::GameState(
  std::uint_fast8_t const room, bool const dong_feng_zhan, Kanachan::Deciders deciders,
  std::array<std::uint_fast8_t, 4u> const &grades, std::stop_token stop_token)
  : room_(room)
  , dong_feng_zhan_(dong_feng_zhan)
  , deciders_(deciders)
  , grades_(grades)
  , stop_token_(stop_token)
{
  KANACHAN_ASSERT((room_ < 5u));
  for (auto const &decider : deciders) {
    if (!decider) {
      KANACHAN_THROW<std::invalid_argument>("One of `deciders` is empty.");
    }
  }
}

std::uint_fast8_t GameState::getRoom() const
{
  return room_;
}

bool GameState::isDongfengZhan() const
{
  return dong_feng_zhan_;
}

std::uint_fast8_t GameState::getPlayerGrade(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  return grades_[seat];
}

std::uint_fast8_t GameState::getChang() const
{
  KANACHAN_ASSERT((dong_feng_zhan_ && chang_ < 2u || !dong_feng_zhan_ && chang_ < 3u));
  return chang_;
}

std::uint_fast8_t GameState::getJu() const
{
  KANACHAN_ASSERT((ju_ < 4u));
  return ju_;
}

std::uint_fast8_t GameState::getBenChang() const
{
  return ben_chang_;
}

std::uint_fast8_t GameState::getNumLizhiDeposits() const
{
  return lizhi_deposits_;
}

std::int_fast32_t GameState::getPlayerScore(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  return scores_[seat];
}

std::uint_fast8_t GameState::getPlayerRanking(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  std::int_fast32_t const score = scores_[seat];
  std::uint_fast8_t ranking = 0u;
  for (std::uint_fast8_t i = 0u; i < seat; ++i) {
    if (scores_[i] >= score) {
      ++ranking;
    }
  }
  for (std::uint_fast8_t i = seat + 1u; i < 4u; ++i) {
    if (scores_[i] > score) {
      ++ranking;
    }
  }
  return ranking;
}

std::uint_fast16_t GameState::selectAction(
  std::uint_fast8_t const seat, std::vector<std::uint_fast16_t> &&sparse,
  std::vector<std::uint_fast32_t> &&numeric, std::vector<std::uint_fast16_t> &&progression,
  std::vector<std::uint_fast16_t> &&candidates, Kanachan::GameLog &game_log) const
{
  KANACHAN_ASSERT((seat < 4u));
  return deciders_[seat](
    seat, std::move(sparse), std::move(numeric), std::move(progression), std::move(candidates),
    stop_token_, game_log);
}

void GameState::onSuccessfulLizhi(std::uint_fast8_t const seat)
{
  KANACHAN_ASSERT((seat < 4u));
  KANACHAN_ASSERT((scores_[seat] >= 1000));
  scores_[seat] -= 1000;
  ++lizhi_deposits_;
}

void GameState::addPlayerScore(std::uint_fast8_t const seat, std::int_fast32_t const score)
{
  scores_[seat] += score;
}

void GameState::onLianzhuang(RoundEndStatus const round_end_status)
{
  ++ben_chang_;
  if (round_end_status == RoundEndStatus::hule) {
    lizhi_deposits_ = 0u;
  }
}

void GameState::onLunzhuang(RoundEndStatus const round_end_status)
{
  if (round_end_status == RoundEndStatus::liuju) {
    KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
  }

  if (++ju_ == 4u) {
    ++chang_;
    ju_ = 0u;
  }

  switch (round_end_status) {
  case RoundEndStatus::hule:
    ben_chang_ = 0u;
    lizhi_deposits_ = 0u;
    break;
  case RoundEndStatus::huangpai_pingju:
    ++ben_chang_;
    break;
  default:
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
}

} // namespace Kanachan
