#include "simulation/huangpai_pingju.hpp"

#include "simulation/game_log.hpp"
#include "simulation/round_state.hpp"
#include "common/throw.hpp"
#include <any>
#include <stdexcept>


namespace Kanachan{

std::any huangpaiPingju(Kanachan::RoundState &round_state, Kanachan::GameLog &game_log)
{
  return round_state.onHuangpaiPingju(game_log);
}

} // namespace Kanachan
