#include "simulation/sigang_sanle.hpp"

#include "simulation/game_log.hpp"
#include "simulation/round_state.hpp"


namespace Kanachan{

bool sigangSanle(Kanachan::RoundState &round_state, Kanachan::GameLog &game_log)
{
  round_state.onLiuju(game_log);
  return false;
}

} // namespace Kanachan
