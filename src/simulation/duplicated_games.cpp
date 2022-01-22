#include "simulation/duplicated_games.hpp"

#include "simulation/game.hpp"
#include "simulation/paishan.hpp"
#include "simulation/model_wrapper.hpp"
#include "simulation/utility.hpp"
#include "common/assert.hpp"
#include <boost/python/dict.hpp>
#include <boost/python/object.hpp>
#include <random>
#include <vector>
#include <array>
#include <utility>
#include <cstdint>


namespace Kanachan{

namespace python = boost::python;

void simulateDuplicatedGames(
  std::uint_fast8_t room, bool dong_feng_zhan, bool one_versus_three,
  std::uint_fast8_t const baseline_grade, Kanachan::ModelWrapper baseline_model,
  std::uint_fast8_t const proposed_grade, Kanachan::ModelWrapper proposed_model,
  std::vector<Kanachan::Paishan> const &test_paishan_list, python::dict result)
{
  KANACHAN_ASSERT((room < 5u));
  KANACHAN_ASSERT((baseline_grade < 16u));
  KANACHAN_ASSERT((!baseline_model.is_none()));
  KANACHAN_ASSERT((proposed_grade < 16u));
  KANACHAN_ASSERT((!proposed_model.is_none()));
  KANACHAN_ASSERT((!result.is_none()));

  std::vector<std::uint_least32_t> seed = Kanachan::getRandomSeed();

  auto game_simulator = [&](std::array<bool, 4u> const &flags) -> void
  {
    using Seat = std::pair<std::uint_fast8_t, Kanachan::ModelWrapper>;
    std::array<Seat, 4u> seats = {
      flags[0u] ? Seat(proposed_grade, proposed_model) : Seat(baseline_grade, baseline_model),
      flags[1u] ? Seat(proposed_grade, proposed_model) : Seat(baseline_grade, baseline_model),
      flags[2u] ? Seat(proposed_grade, proposed_model) : Seat(baseline_grade, baseline_model),
      flags[3u] ? Seat(proposed_grade, proposed_model) : Seat(baseline_grade, baseline_model)
    };

    std::seed_seq ss(seed.cbegin(), seed.cend());
    std::mt19937 urng(ss);
    python::dict game_result = Kanachan::simulateGame(
      urng, room, dong_feng_zhan, seats, test_paishan_list);

    for (long i = 0; i < python::len(game_result["rounds"]); ++i) {
      python::object round = game_result["rounds"][i];
      KANACHAN_ASSERT((python::len(round) == 4));
      for (long j = 0; j < 4; ++j) {
        python::object e = round[j];
        if (flags[j]) {
          result["proposed"]["rounds"].attr("append")(e);
        }
        else {
          result["baseline"]["rounds"].attr("append")(e);
        }
      }
    }
    KANACHAN_ASSERT((python::len(game_result["final_ranking"]) == 4));
    KANACHAN_ASSERT((python::len(game_result["final_scores"]) == 4));
    for (long i = 0; i < 4; ++i) {
      python::dict game;
      game["ranking"] = game_result["final_ranking"][i];
      game["score"] = game_result["final_scores"][i];
      if (flags[i]) {
        result["proposed"]["games"].attr("append")(game);
      }
      else {
        result["baseline"]["games"].attr("append")(game);
      }
    }
  };

  if (one_versus_three) {
    game_simulator({false, false, false, true});
    game_simulator({false, false, true, false});
    game_simulator({false, true, false, false});
    game_simulator({true, false, false, false});
    return;
  }

  game_simulator({false, false, true, true});
  game_simulator({false, true, false, true});
  game_simulator({false, true, true, false});
  game_simulator({true, false, false, true});
  game_simulator({true, false, true, false});
  game_simulator({true, true, false, false});
}

} // namespace Kanachan
