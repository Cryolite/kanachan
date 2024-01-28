#include "simulation/game_log.hpp"

#include "simulation/gil.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/import.hpp>
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

GameLog::GameLog()
  : data_()
  , round_results_()
  , game_scores_({ INT_FAST32_MAX, INT_FAST32_MAX, INT_FAST32_MAX, INT_FAST32_MAX })
  , with_proposed_model_()
{}

void GameLog::onBeginningOfRound()
{
  // Do nothing.
}

void GameLog::onDecision(std::uint_fast8_t seat, python::object data)
{
  KANACHAN_ASSERT((seat < 4u));
  data_[seat].append(data);
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
      round_result_["chang"] = round_result.getChang();
      round_result_["ju"] = round_result.getJu();
      round_result_["benchang"] = round_result.getBenChang();
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

  python::object torch = python::import("torch");

  python::dict result;
  python::list episode;
  result["proposed"] = with_proposed_model_[seat];
  result["episode"] = torch.attr("stack")(data_[seat]);
  result["scores"] = python::list();
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    result["scores"].attr("append")(game_scores_[i]);
  }

  return result;
}

}
