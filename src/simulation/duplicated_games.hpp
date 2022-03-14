#if !defined(KANACHAN_SIMULATION_DUPLICATED_GAMES_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_DUPLICATED_GAMES_HPP_INCLUDE_GUARD

#include "simulation/paishan.hpp"
#include "simulation/model_wrapper.hpp"
#include <boost/python/list.hpp>
#include <vector>
#include <cstdint>


namespace Kanachan{

boost::python::list simulateDuplicatedGames(
  std::uint_fast8_t room, bool dong_feng_zhan, bool one_versus_three,
  std::uint_fast8_t baseline_grade, Kanachan::ModelWrapper baseline_model,
  std::uint_fast8_t proposed_grade, Kanachan::ModelWrapper proposed_model,
  std::vector<Kanachan::Paishan> const &test_paishan_list);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_DUPLICATED_GAMES_HPP_INCLUDE_GUARD)
