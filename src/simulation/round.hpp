#if !defined(KANACHAN_SIMULATION_ROUND_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_ROUND_HPP_INCLUDE_GUARD

#include "simulation/paishan.hpp"
#include "simulation/game_log.hpp"
#include "simulation/game_state.hpp"
#include <random>
#include <vector>
#include <cstdint>


namespace Kanachan{

bool simulateRound(
  std::mt19937 &urng, Kanachan::GameState &game_state, Kanachan::Paishan const *p_test_paishan,
  Kanachan::GameLog &game_log);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_ROUND_HPP_INCLUDE_GUARD)
