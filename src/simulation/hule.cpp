#include "simulation/hule.hpp"

#include "simulation/game_log.hpp"
#include "simulation/round_state.hpp"
#include "common/throw.hpp"
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>


namespace{

using std::placeholders::_1;

} // namespace `anonymous`

namespace Kanachan{

bool hule(
  Kanachan::RoundState &round_state, std::uint_fast8_t const zimo_tile,
  Kanachan::GameLog &game_log)
{
  if (zimo_tile >= 37u && zimo_tile != UINT_FAST8_MAX) {
    KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(zimo_tile);
  }

  return round_state.onHule(zimo_tile, game_log);
}

} // namespace Kanachan
