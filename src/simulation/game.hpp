#if !defined(KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD

#include "simulation/game_log.hpp"
#include "simulation/paishan.hpp"
#include "simulation/decision_maker.hpp"
#include <boost/python/dict.hpp>
#include <stop_token>
#include <random>
#include <vector>
#include <array>
#include <utility>
#include <memory>
#include <cstdint>


namespace Kanachan{

std::shared_ptr<Kanachan::GameLog> simulateGame(
  std::mt19937 &urng, std::uint_fast8_t room, bool dong_feng_zhan, Kanachan::Deciders deciders,
  std::array<std::uint_fast8_t, 4u> const &grades,
  std::vector<Kanachan::Paishan> const &test_paishan_list, std::stop_token stop_token);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD)
