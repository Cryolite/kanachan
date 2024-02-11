#if !defined(KANACHAN_SIMULATION_DAPAI_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_DAPAI_HPP_INCLUDE_GUARD

#include "simulation/game_log.hpp"
#include "simulation/round_state.hpp"
#include <any>
#include <cstdint>


namespace Kanachan{

std::any dapai(
  Kanachan::RoundState &round_state, std::uint_fast8_t tile, bool moqi, bool lizhi,
  Kanachan::GameLog &game_log);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_DAPAI_HPP_INCLUDE_GUARD)
