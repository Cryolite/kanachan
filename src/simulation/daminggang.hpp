#if !defined(KANACHAN_SIMULATION_DAMINGGANG_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_DAMINGGANG_HPP_INCLUDE_GUARD

#include "simulation/game_log.hpp"
#include "simulation/round_state.hpp"
#include <any>


namespace Kanachan{

std::any daminggang(Kanachan::RoundState &round_state, Kanachan::GameLog &game_log);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_DAMINGGANG_HPP_INCLUDE_GUARD)
