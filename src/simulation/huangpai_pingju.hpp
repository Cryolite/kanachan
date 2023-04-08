#if !defined(KANACHAN_SIMULATION_HUANGPAI_PINGJU_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_HUANGPAI_PINGJU_HPP_INCLUDE_GUARD

#include "simulation/round_state.hpp"
#include <boost/python/dict.hpp>
#include <any>


namespace Kanachan{

std::any huangpaiPingju(Kanachan::RoundState &round_state, boost::python::dict result);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_HUANGPAI_PINGJU_HPP_INCLUDE_GUARD)
