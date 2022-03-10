#include "simulation/duplicated_games.hpp"

#include "simulation/game.hpp"
#include "simulation/paishan.hpp"
#include "simulation/model_wrapper.hpp"
#include "simulation/utility.hpp"
#include "common/assert.hpp"
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <random>
#include <vector>
#include <array>
#include <utility>
#include <cstdint>


namespace Kanachan{

namespace python = boost::python;

python::list simulateDuplicatedGames(
  std::uint_fast8_t room, bool dong_feng_zhan, bool one_versus_three,
  std::uint_fast8_t const baseline_grade, Kanachan::ModelWrapper baseline_model,
  std::uint_fast8_t const proposed_grade, Kanachan::ModelWrapper proposed_model,
  std::vector<Kanachan::Paishan> const &test_paishan_list)
{
  KANACHAN_ASSERT((room < 5u));
  KANACHAN_ASSERT((baseline_grade < 16u));
  KANACHAN_ASSERT((!baseline_model.is_none()));
  KANACHAN_ASSERT((proposed_grade < 16u));
  KANACHAN_ASSERT((!proposed_model.is_none()));

  std::vector<std::uint_least32_t> seed = Kanachan::getRandomSeed();

  auto game_simulator = [&](std::array<bool, 4u> const &flags) -> python::dict
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
    python::dict result = Kanachan::simulateGame(
      urng, room, dong_feng_zhan, seats, test_paishan_list);

    result["proposed"] = python::list();
    result["proposed"].attr("append")(flags[0u] ? 1 : 0);
    result["proposed"].attr("append")(flags[1u] ? 1 : 0);
    result["proposed"].attr("append")(flags[2u] ? 1 : 0);
    result["proposed"].attr("append")(flags[3u] ? 1 : 0);

    return result;
  };

  if (one_versus_three) {
    python::list results;
    python::dict result = game_simulator({false, false, false, true});
    results.attr("append")(result);
    result = game_simulator({false, false, true, false});
    results.attr("append")(result);
    result = game_simulator({false, true, false, false});
    results.attr("append")(result);
    result = game_simulator({true, false, false, false});
    results.attr("append")(result);
    return results;
  }

  python::list results;
  python::dict result = game_simulator({false, false, true, true});
  results.attr("append")(result);
  result = game_simulator({false, true, false, true});
  results.attr("append")(result);
  result = game_simulator({false, true, true, false});
  results.attr("append")(result);
  result = game_simulator({true, false, false, true});
  results.attr("append")(result);
  result = game_simulator({true, false, true, false});
  results.attr("append")(result);
  result = game_simulator({true, true, false, false});
  results.attr("append")(result);
  return results;
}

} // namespace Kanachan
