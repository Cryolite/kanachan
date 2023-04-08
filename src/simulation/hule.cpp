#include "simulation/hule.hpp"

#include "simulation/round_state.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>


namespace{

using std::placeholders::_1;
namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

bool hule(Kanachan::RoundState &round_state, std::uint_fast8_t const zimo_tile, python::dict result)
{
  if (zimo_tile >= 37u && zimo_tile != std::numeric_limits<std::uint8_t>::max()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(zimo_tile);
  }
  if (result.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`result` must not be a `None`.");
  }

  return round_state.onHule(zimo_tile, result);
}

} // namespace Kanachan
