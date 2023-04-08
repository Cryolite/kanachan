#include "simulation/round.hpp"

#include "simulation/zimo.hpp"
#include "simulation/round_state.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <vector>
#include <functional>
#include <any>
#include <stdexcept>
#include <cstdint>


namespace{

namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

bool simulateRound(
  std::vector<std::uint_least32_t> const &seed, Kanachan::GameState &game_state,
  Kanachan::Paishan const * const p_test_paishan, python::dict result)
{
  if (result.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`result` must not be a `None`.");
  }

  Kanachan::RoundState round_state(seed, game_state, p_test_paishan);

  std::function<std::any()> next_step = std::bind(&Kanachan::zimo, std::ref(round_state), result);

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
