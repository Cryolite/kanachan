#include "simulation/round.hpp"

#include "simulation/zimo.hpp"
#include "simulation/round_state.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include "simulation/game_log.hpp"
#include "common/throw.hpp"
#include <random>
#include <vector>
#include <functional>
#include <any>
#include <stdexcept>
#include <cstdint>


namespace Kanachan{

bool simulateRound(
  std::mt19937 &urng, Kanachan::GameState &game_state,
  Kanachan::Paishan const * const p_test_paishan, Kanachan::GameLog &game_log)
{
  Kanachan::RoundState round_state(urng, game_state, p_test_paishan);
  game_log.onBeginningOfRound();

  std::function<std::any()> next_step = std::bind(
    &Kanachan::zimo, std::ref(round_state), std::ref(game_log));

  for (;;) {
    std::any next_step_ = next_step();
    if (std::any_cast<std::function<std::any()>>(&next_step_) != nullptr) {
      next_step = std::any_cast<std::function<std::any()>>(next_step_);
      continue;
    }
    if (std::any_cast<bool>(&next_step_) != nullptr) {
      return std::any_cast<bool>(next_step_);
    }
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
}

} // namespace Kanachan
