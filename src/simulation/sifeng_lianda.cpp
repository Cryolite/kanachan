#include "simulation/sifeng_lianda.hpp"

#include "simulation/game_log.hpp"
#include "simulation/round_state.hpp"
#include "common/throw.hpp"
#include <stdexcept>


namespace Kanachan{

bool sifengLianda(Kanachan::RoundState &round_state, Kanachan::GameLog &game_log)
{
  round_state.onLiuju(game_log);
  return false;
}

} // namespace Kanachan
