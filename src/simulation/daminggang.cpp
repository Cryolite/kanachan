#include "simulation/daminggang.hpp"

#include "simulation/zimo.hpp"
#include "simulation/game_log.hpp"
#include "simulation/round_state.hpp"
#include <any>
#include <utility>


namespace Kanachan{

std::any daminggang(Kanachan::RoundState &round_state, Kanachan::GameLog &game_log)
{
  round_state.onDaminggang(game_log);

  // Si Gang San Le (四槓散了) の成立は打牌直後．

  // Zimo (自摸)
  auto zimo = std::bind(&Kanachan::zimo, std::ref(round_state), std::ref(game_log));
  std::function<std::any()> next_step(std::move(zimo));
  return next_step;
}

} // namespace Kanachan
