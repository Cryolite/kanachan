#if !defined(KANACHAN_SIMULATION_DUPLICATED_GAMES_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_DUPLICATED_GAMES_HPP_INCLUDE_GUARD

#include "simulation/paishan.hpp"
#include <boost/python/dict.hpp>
#include <boost/python/object.hpp>
#include <vector>
#include <cstdint>


namespace Kanachan{

void simulateDuplicatedGames(
  std::uint_fast8_t room, bool dong_feng_zhan, bool one_versus_three,
  std::uint_fast8_t baseline_grade, boost::python::object baseline_model,
  std::uint_fast8_t proposal_grade, boost::python::object proposal_model,
  boost::python::object external_tool,
  std::vector<Kanachan::Paishan> const &test_paishan_list,
  boost::python::dict result);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_DUPLICATED_GAMES_HPP_INCLUDE_GUARD)
