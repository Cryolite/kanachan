#include "simulation/game_log.hpp"

#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <vector>
#include <functional>
#include <cstdint>


namespace{

using std::placeholders::_1;
namespace python = boost::python;

}

namespace Kanachan{

GameLog::Decision_::Decision_(
  std::vector<std::uint_fast16_t> &&sparse, std::vector<std::uint_fast16_t> &&numeric,
  std::vector<std::uint_fast16_t> &&progression, std::vector<std::uint_fast16_t> &&candidates,
  std::uint_fast16_t const action)
  : sparse_(std::move(sparse))
  , numeric_(std::move(numeric))
  , progression_(std::move(progression))
  , candidates_(std::move(candidates))
  , action_index_(UINT_FAST8_MAX)
{
  for (std::uint_fast8_t i = 0u; i < candidates_.size(); ++i) {
    if (candidates_[i] == action) {
      action_index_ = i;
      break;
    }
  }
  if (action_index_ == UINT_FAST8_MAX) {
    KANACHAN_THROW<std::logic_error>(_1) << action;
  }
}

std::vector<std::uint_fast16_t> const &GameLog::Decision_::getSparse() const noexcept
{
  return sparse_;
}

std::vector<std::uint_fast16_t> const &GameLog::Decision_::getNumeric() const noexcept
{
  return numeric_;
}

std::vector<std::uint_fast16_t> const &GameLog::Decision_::getProgression() const noexcept
{
  return progression_;
}

std::vector<std::uint_fast16_t> const &GameLog::Decision_::getCandidates() const noexcept
{
  return candidates_;
}

std::uint_fast8_t GameLog::Decision_::getActionIndex() const noexcept
{
  return action_index_;
}

GameLog::GameLog()
  : decisions_()
  , round_results_()
  , game_scores_({ INT_FAST32_MAX, INT_FAST32_MAX, INT_FAST32_MAX, INT_FAST32_MAX })
{}

void GameLog::onBeginningOfRound()
{
  // Do nothing.
}

void GameLog::onDecision(
  std::uint_fast8_t seat, std::vector<std::uint_fast16_t> &&sparse,
  std::vector<std::uint_fast16_t> &&numeric, std::vector<std::uint_fast16_t> &&progression,
  std::vector<std::uint_fast16_t> &&candidates, std::uint_fast16_t action)
{
  KANACHAN_ASSERT((seat < 4u));
  KANACHAN_ASSERT((action < 546u));

  decisions_[seat].emplace_back(
    std::move(sparse), std::move(numeric), std::move(progression), std::move(candidates), action);
}

void GameLog::onEndOfRound(std::array<Kanachan::RoundResult, 4u> const &round_results)
{
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    round_results_[i].emplace_back(round_results[i]);
  }
}

void GameLog::onEndOfGame(std::array<std::int_fast32_t, 4u> const &scores)
{
  game_scores_ = scores;
}

void GameLog::setWithProposedModel(std::array<bool, 4u> const &with_proposed_model)
{
  with_proposed_model_ = with_proposed_model;
}

python::list GameLog::getResult() const
{
  python::list game_results = python::list();
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    python::dict game_result;
    game_results.append(game_result);
    game_result["proposed"] = with_proposed_model_[i];
    game_result["score"] = game_scores_[i];

    python::list round_results;
    game_result["round_results"] = round_results;
    for (auto const &round_result : round_results_[i]) {
      python::dict round_result_;
      round_results.append(round_result_);
      round_result_["type"] = round_result.getType();
      round_result_["in_lizhi"] = round_result.getInLizhi();
      round_result_["has_fulu"] = round_result.getHasFulu();
      round_result_["delta_score"] = round_result.getRoundDeltaScore();
      round_result_["score"] = round_result.getRoundScore();
    }

    std::uint_fast8_t ranking = 0u;
    for (std::uint_fast8_t j = 0u; j < i; ++j) {
      if (game_scores_[j] >= game_scores_[i]) {
        ++ranking;
      }
    }
    for (std::uint_fast8_t j = i + 1u; j < 4u; ++j) {
      if (game_scores_[j] > game_scores_[i]) {
        ++ranking;
      }
    }
    game_result["ranking"] = ranking;
  }

  return game_results;
}

python::dict GameLog::getEpisode(std::uint_fast8_t const seat) const
{
  if (seat >= 4u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(seat);
  }

  python::dict result;
  python::list episode;
  result["proposed"] = with_proposed_model_[seat];
  result["episode"] = episode;
  for (auto const &decision : decisions_[seat]) {
    python::dict d;
    episode.append(d);

    python::list sparse;
    d["sparse"] = sparse;
    for (auto f : decision.getSparse()) {
      sparse.append(f);
    }
    while (python::len(sparse) < 33) {
      sparse.append(526);
    }

    python::list numeric;
    d["numeric"] = numeric;
    numeric.append(static_cast<double>(decision.getNumeric()[0u]));
    numeric.append(static_cast<double>(decision.getNumeric()[1u]));
    numeric.append(static_cast<double>(decision.getNumeric()[2u]) / 10000.0);
    numeric.append(static_cast<double>(decision.getNumeric()[3u]) / 10000.0);
    numeric.append(static_cast<double>(decision.getNumeric()[4u]) / 10000.0);
    numeric.append(static_cast<double>(decision.getNumeric()[5u]) / 10000.0);

    python::list progression;
    d["progression"] = progression;
    for (auto f : decision.getProgression()) {
      progression.append(f);
    }
    while (python::len(progression) < 113) {
      progression.append(2165);
    }

    python::list candidates;
    d["candidates"] = candidates;
    for (auto f : decision.getCandidates()) {
      candidates.append(f);
    }
    candidates.append(546);
    while (python::len(candidates) < 32) {
      candidates.append(547);
    }

    d["action_index"] = decision.getActionIndex();
  }
  result["score"] = game_scores_[seat];

  std::uint_fast8_t ranking = 0u;
  for (std::uint_fast8_t i = 0u; i < seat; ++i) {
    if (game_scores_[i] >= game_scores_[seat]) {
      ++ranking;
    }
  }
  for (std::uint_fast8_t i = seat + 1u; i < 4u; ++i) {
    if (game_scores_[i] > game_scores_[seat]) {
      ++ranking;
    }
  }
  result["ranking"] = ranking;

  return result;
}

}
