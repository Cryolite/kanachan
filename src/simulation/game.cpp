#include "simulation/game.hpp"

#include "simulation/game_log.hpp"
#include "simulation/round.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include "simulation/decision_maker.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <stop_token>
#include <vector>
#include <array>
#include <functional>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <cstddef>


namespace {

using std::placeholders::_1;
using Seat_ = std::pair<std::uint8_t, std::shared_ptr<Kanachan::DecisionMaker>>;
using Seats_ = std::array<Seat_, 4u>;

} // namespace `anonymous`

namespace Kanachan{

std::shared_ptr<Kanachan::GameLog> simulateGame(
  std::vector<std::uint_least32_t> const &seed, std::uint_fast8_t room, bool dong_feng_zhan,
  Seats_ const &seats, std::vector<Kanachan::Paishan> const &test_paishan_list,
  std::stop_token stop_token)
{
  if (seed.empty() && test_paishan_list.empty()) {
    KANACHAN_THROW<std::invalid_argument>(
      "Either `seed` or `test_paishan_list` must not be empty.");
  }
  if (!seed.empty() && !test_paishan_list.empty()) {
    KANACHAN_THROW<std::invalid_argument>("Either `seed` or `test_paishan_list` must be empty.");
  }
  if (room >= 5u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << room;
  }
  for (auto [grade, model] : seats) {
    if (grade >= 16u) {
      KANACHAN_THROW<std::invalid_argument>(_1) << grade;
    }
    if (!model) {
      KANACHAN_THROW<std::invalid_argument>("The model of a seat is empty.");
    }
  }

  bool const test = !test_paishan_list.empty();
  std::size_t i = 0u;

  Kanachan::GameState game_state(room, dong_feng_zhan, seats, stop_token);
  std::shared_ptr<Kanachan::GameLog> p_game_log = std::make_shared<Kanachan::GameLog>();

  bool end_of_game = false;
  while (!end_of_game) {
    if (test) {
      if (i >= test_paishan_list.size()) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << "The number of test PaiShan is too small: i == " << i
          << ", test_paishan_list.size() == " << test_paishan_list.size();
      }
      Kanachan::Paishan const &test_paishan = test_paishan_list[i++];
      end_of_game = Kanachan::simulateRound(seed, game_state, &test_paishan, *p_game_log);
      if (end_of_game && i != test_paishan_list.size()) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << "The number of test pai shan is too large: i == " << i
          << ", test_paishan_list.size() == " << test_paishan_list.size();
      }
    }
    else {
      end_of_game = Kanachan::simulateRound(seed, game_state, nullptr, *p_game_log);
    }
  }

  p_game_log->onEndOfGame({
    game_state.getPlayerScore(0u), game_state.getPlayerScore(1u),
    game_state.getPlayerScore(2u), game_state.getPlayerScore(3u)
  });

  return p_game_log;
}

} // namespace Kanachan
