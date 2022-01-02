#include "simulation/duplicated_games.hpp"

#include "simulation/game.hpp"
#include "simulation/paishan.hpp"
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
  std::uint_fast8_t const baseline_grade, python::object baseline_model,
  std::uint_fast8_t const proposal_grade, python::object proposal_model,
  python::object external_tool,
  std::vector<Kanachan::Paishan> const &test_paishan_list, python::dict result)
{
  KANACHAN_ASSERT((room < 5u));
  KANACHAN_ASSERT((baseline_grade < 16u));
  KANACHAN_ASSERT((!baseline_model.is_none()));
  KANACHAN_ASSERT((proposal_grade < 16u));
  KANACHAN_ASSERT((!proposal_model.is_none()));
  KANACHAN_ASSERT((!external_tool.is_none()));
  KANACHAN_ASSERT((!result.is_none()));

  using Seat = std::pair<std::uint_fast8_t, python::object>;

  std::vector<std::uint_least32_t> seed = Kanachan::getRandomSeed();

  auto game_simulator = [&](std::array<Seat, 4u> const &seats) {
    std::seed_seq ss(seed.cbegin(), seed.cend());
    std::mt19937 urng(ss);
    Kanachan::simulateGame(
      urng, room, dong_feng_zhan, seats, external_tool, test_paishan_list,
      result);
  };

  if (one_versus_three) {
    {
      std::array<Seat, 4u> seats{
        Seat(baseline_grade, baseline_model),
        Seat(baseline_grade, baseline_model),
        Seat(baseline_grade, baseline_model),
        Seat(proposal_grade, proposal_model)
      };
      game_simulator(seats);
    }
    {
      std::array<Seat, 4u> seats{
        Seat(baseline_grade, baseline_model),
        Seat(baseline_grade, baseline_model),
        Seat(proposal_grade, proposal_model),
        Seat(baseline_grade, baseline_model)
      };
      game_simulator(seats);
    }
    {
      std::array<Seat, 4u> seats{
        Seat(baseline_grade, baseline_model),
        Seat(proposal_grade, proposal_model),
        Seat(baseline_grade, baseline_model),
        Seat(baseline_grade, baseline_model)
      };
      game_simulator(seats);
    }
    {
      std::array<Seat, 4u> seats{
        Seat(proposal_grade, proposal_model),
        Seat(baseline_grade, baseline_model),
        Seat(baseline_grade, baseline_model),
        Seat(baseline_grade, baseline_model)
      };
      game_simulator(seats);
    }
    return;
  }

  {
    std::array<Seat, 4u> seats{
      Seat(baseline_grade, baseline_model),
      Seat(baseline_grade, baseline_model),
      Seat(proposal_grade, proposal_model),
      Seat(proposal_grade, proposal_model)
    };
    game_simulator(seats);
  }
  {
    std::array<Seat, 4u> seats{
      Seat(baseline_grade, baseline_model),
      Seat(proposal_grade, proposal_model),
      Seat(baseline_grade, baseline_model),
      Seat(proposal_grade, proposal_model)
    };
    game_simulator(seats);
  }
  {
    std::array<Seat, 4u> seats{
      Seat(baseline_grade, baseline_model),
      Seat(proposal_grade, proposal_model),
      Seat(proposal_grade, proposal_model),
      Seat(baseline_grade, baseline_model)
    };
    game_simulator(seats);
  }
  {
    std::array<Seat, 4u> seats{
      Seat(proposal_grade, proposal_model),
      Seat(baseline_grade, baseline_model),
      Seat(baseline_grade, baseline_model),
      Seat(proposal_grade, proposal_model)
    };
    game_simulator(seats);
  }
  {
    std::array<Seat, 4u> seats{
      Seat(proposal_grade, proposal_model),
      Seat(baseline_grade, baseline_model),
      Seat(proposal_grade, proposal_model),
      Seat(baseline_grade, baseline_model)
    };
    game_simulator(seats);
  }
  {
    std::array<Seat, 4u> seats{
      Seat(proposal_grade, proposal_model),
      Seat(proposal_grade, proposal_model),
      Seat(baseline_grade, baseline_model),
      Seat(baseline_grade, baseline_model)
    };
    game_simulator(seats);
  }
}

} // namespace Kanachan
