#include "simulation/round.hpp"

#include "simulation/zimo.hpp"
#include "simulation/round_state.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <random>
#include <functional>
#include <any>
#include <stdexcept>


namespace Kanachan{

namespace python = boost::python;

bool simulateRound(
  std::mt19937 &urng, Kanachan::GameState &game_state,
  Kanachan::Paishan const *p_test_paishan, python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));

  Kanachan::RoundState round_state(urng, game_state, p_test_paishan);

  std::function<std::any()> next_step = std::bind(
    &Kanachan::zimo, std::ref(round_state), result);

  for (;;) {
    std::any next_step_ = next_step();
    if (std::any_cast<std::function<std::any()> >(&next_step_) != nullptr) {
      next_step = std::any_cast<std::function<std::any()> >(next_step_);
      continue;
    }
    if (std::any_cast<bool>(&next_step_) != nullptr) {
      return std::any_cast<bool>(next_step_);
    }
    KANACHAN_THROW<std::logic_error>("");
  }
}

} // namespace Kanachan
