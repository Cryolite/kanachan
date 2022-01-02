#if !defined(KANACHAN_SIMULATION_SIGANG_SANLE_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_SIGANG_SANLE_HPP_INCLUDE_GUARD

#include "simulation/round_state.hpp"
#include <boost/python/dict.hpp>


namespace Kanachan{

bool sigangSanle(Kanachan::RoundState &round_state, boost::python::dict result);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_SIGANG_SANLE_HPP_INCLUDE_GUARD)
