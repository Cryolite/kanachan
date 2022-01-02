#include "simulation/huangpai_pingju.hpp"

#include "simulation/round_state.hpp"
#include "common/assert.hpp"
#include <boost/python/dict.hpp>
#include <any>


namespace Kanachan{

namespace python = boost::python;

std::any huangpaiPingju(Kanachan::RoundState &round_state, python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));

  return round_state.onHuangpaiPingju(result);
}

} // namespace Kanachan
