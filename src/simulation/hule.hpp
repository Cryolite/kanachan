#if !defined(KANACHAN_SIMULATION_HULE_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_HULE_HPP_INCLUDE_GUARD

#include "simulation/game_log.hpp"
#include "simulation/round_state.hpp"
#include <cstdint>


namespace Kanachan{

bool hule(
  Kanachan::RoundState &round_state, std::uint_fast8_t zimo_tile, Kanachan::GameLog &game_log);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_HULE_HPP_INCLUDE_GUARD)
