#include "simulation/sifeng_lianda.hpp"

#include "simulation/round_state.hpp"
#include "common/assert.hpp"
#include <boost/python/dict.hpp>


namespace Kanachan{

namespace python = boost::python;

bool sifengLianda(Kanachan::RoundState &round_state, python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));

  round_state.onLiuju(result);
  return false;
}

} // namespace Kanachan
