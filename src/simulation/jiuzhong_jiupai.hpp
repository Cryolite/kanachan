#if !defined(KANACHNA_SIMULATION_JIUZHONG_JIUPAI_HPP_INCLUDE_GUARD)
#define KANACHNA_SIMULATION_JIUZHONG_JIUPAI_HPP_INCLUDE_GUARD

#include "simulation/round_state.hpp"
#include <boost/python/dict.hpp>


namespace Kanachan{

bool jiuzhongJiupai(Kanachan::RoundState &round_state, boost::python::dict result);

} // namespace Kanachan

#endif // !defined(KANACHNA_SIMULATION_JIUZHONG_JIUPAI_HPP_INCLUDE_GUARD)
