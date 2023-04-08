#include "simulation/sigang_sanle.hpp"

#include "simulation/round_state.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <stdexcept>


namespace{

namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

bool sigangSanle(Kanachan::RoundState &round_state, python::dict result)
{
  if (result.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`result` must not be a `None`.");
  }

  round_state.onLiuju(result);
  return false;
}

} // namespace Kanachan
