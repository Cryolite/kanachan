#include "simulation/hule.hpp"

#include "simulation/round_state.hpp"
#include "common/assert.hpp"
#include <boost/python/dict.hpp>
#include <limits>
#include <cstdint>


namespace Kanachan{

namespace python = boost::python;

bool hule(
  Kanachan::RoundState &round_state, std::uint_fast8_t zimo_tile,
  python::dict result)
{
  KANACHAN_ASSERT(
    (zimo_tile < 37u || zimo_tile == std::numeric_limits<std::uint_fast8_t>::max()));
  KANACHAN_ASSERT((!result.is_none()));

  return round_state.onHule(zimo_tile, result);
}

} // namespace Kanachan
