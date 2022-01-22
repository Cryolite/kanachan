#if !defined(KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD

#include "simulation/paishan.hpp"
#include "simulation/model_wrapper.hpp"
#include <boost/python/dict.hpp>
#include <random>
#include <vector>
#include <array>
#include <utility>
#include <cstdint>


namespace Kanachan{

boost::python::dict simulateGame(
  std::mt19937 &urng, std::uint_fast8_t room, bool dong_feng_zhan,
  std::array<std::pair<std::uint_fast8_t, Kanachan::ModelWrapper>, 4u> const &seats,
  std::vector<Kanachan::Paishan> const &test_paishan_list);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_GAME_HPP_INCLUDE_GUARD)
