#include "simulation/huangpai_pingju.hpp"

#include "simulation/round_state.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <any>
#include <stdexcept>


namespace{

namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

std::any huangpaiPingju(Kanachan::RoundState &round_state, python::dict result)
{
  if (result.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`result` must not be a `None`.");
  }

  return round_state.onHuangpaiPingju(result);
}

} // namespace Kanachan
