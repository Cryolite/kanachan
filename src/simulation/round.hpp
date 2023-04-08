#if !defined(KANACHAN_SIMULATION_ROUND_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_ROUND_HPP_INCLUDE_GUARD

#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include <boost/python/dict.hpp>
#include <vector>
#include <cstdint>


namespace Kanachan{

bool simulateRound(
  std::vector<std::uint_least32_t> const &seed, Kanachan::GameState &game_state,
  Kanachan::Paishan const *p_test_paishan, boost::python::dict result);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_ROUND_HPP_INCLUDE_GUARD)
