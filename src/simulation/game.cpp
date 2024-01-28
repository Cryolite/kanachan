#include "simulation/game.hpp"

#include "simulation/round.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include "simulation/decision_maker.hpp"
#include "simulation/game_log.hpp"
#include "simulation/gil.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <stop_token>
#include <random>
#include <vector>
#include <array>
#include <functional>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <cstddef>


namespace {

using std::placeholders::_1;
namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

std::shared_ptr<Kanachan::GameLog> simulateGame(
  std::mt19937 &urng, std::uint_fast8_t room, bool dong_feng_zhan, Kanachan::Deciders deciders,
  std::array<std::uint_fast8_t, 4u> const &grades,
  std::vector<Kanachan::Paishan> const &test_paishan_list, std::stop_token stop_token)
{
  if (room >= 5u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << room;
  }
  for (auto const &decider : deciders) {
    if (!decider) {
      KANACHAN_THROW<std::invalid_argument>("A decider is empty.");
    }
  }

  bool const test = !test_paishan_list.empty();
  std::size_t i = 0u;

  Kanachan::GameState game_state(room, dong_feng_zhan, deciders, grades, stop_token);
  std::shared_ptr<Kanachan::GameLog> p_game_log;
  {
    Kanachan::GIL::RecursiveLock gil_lock;
    p_game_log = std::make_shared<Kanachan::GameLog>();
  }

  bool end_of_game = false;
  while (!end_of_game) {
    if (test) {
      if (i >= test_paishan_list.size()) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << "The number of test PaiShan is too small: i == " << i
          << ", test_paishan_list.size() == " << test_paishan_list.size();
      }
      Kanachan::Paishan const &test_paishan = test_paishan_list[i++];
      end_of_game = Kanachan::simulateRound(urng, game_state, &test_paishan, *p_game_log);
      if (end_of_game && i != test_paishan_list.size()) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << "The number of test pai shan is too large: i == " << i
          << ", test_paishan_list.size() == " << test_paishan_list.size();
      }
    }
    else {
      end_of_game = Kanachan::simulateRound(urng, game_state, nullptr, *p_game_log);
    }
  }

  p_game_log->onEndOfGame({
    game_state.getPlayerScore(0u), game_state.getPlayerScore(1u),
    game_state.getPlayerScore(2u), game_state.getPlayerScore(3u)
  });

  return p_game_log;
}

} // namespace Kanachan
