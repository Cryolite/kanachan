#if !defined(KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD

#include "simulation/paishan.hpp"
#include "simulation/decision_maker.hpp"
#include <boost/python/dict.hpp>
#include <stop_token>
#include <vector>
#include <array>
#include <utility>
#include <memory>
#include <cstdint>


namespace Kanachan{

boost::python::dict simulateGame(
  std::vector<std::uint_least32_t> const &seed, std::uint_fast8_t room, bool dong_feng_zhan,
  std::array<std::pair<std::uint_fast8_t, std::shared_ptr<Kanachan::DecisionMaker>>, 4u> const &seats,
  std::vector<Kanachan::Paishan> const &test_paishan_list, std::stop_token stop_token);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD)
