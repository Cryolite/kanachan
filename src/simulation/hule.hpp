#if !defined(KANACHAN_SIMULATION_HULE_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_HULE_HPP_INCLUDE_GUARD

#include "simulation/round_state.hpp"
#include <boost/python/dict.hpp>
#include <cstdint>


namespace Kanachan{

bool hule(
  Kanachan::RoundState &round_state, std::uint_fast8_t zimo_tile,
  boost::python::dict result);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_HULE_HPP_INCLUDE_GUARD)
