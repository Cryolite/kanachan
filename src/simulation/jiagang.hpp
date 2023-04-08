#if !defined(KANACHAN_SIMULATION_JIAGANG_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_JIAGANG_HPP_INCLUDE_GUARD

#include "simulation/round_state.hpp"
#include <boost/python/dict.hpp>
#include <any>
#include <cstdint>


namespace Kanachan{

std::any jiagang(
  Kanachan::RoundState &round_state, std::uint_fast8_t tile, std::uint_fast8_t encode,
  boost::python::dict result);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_JIAGANG_HPP_INCLUDE_GUARD)
