#include "simulation/angang.hpp"

#include "simulation/hule.hpp"
#include "simulation/zimo.hpp"
#include "simulation/round_state.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <functional>
#include <any>
#include <utility>
#include <stdexcept>
#include <limits>
#include <cstdint>


namespace {

using std::placeholders::_1;
namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

std::any angang(
  Kanachan::RoundState &round_state, std::uint_fast8_t const zimo_tile,
  std::uint_fast8_t const encode, python::dict result)
{
  if (zimo_tile >= 37u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(zimo_tile);
  }
  if (encode >= 34u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(encode);
  }
  if (result.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`result` must not be a `None`.");
  }

  std::uint_fast16_t const action = round_state.onAngang(zimo_tile, encode);

  if (action == std::numeric_limits<std::uint_fast16_t>::max()) {
    // Si Gang San Le (四槓散了) の成立は打牌直後．
    auto zimo = std::bind(&Kanachan::zimo, std::ref(round_state), result);
    std::function<std::any()> next_step(std::move(zimo));
    return next_step;
  }

  if (action == 543u) {
    // Qiang Gang (槍槓)
    std::uint_fast8_t const zimo_tile_ = std::numeric_limits<std::uint_fast8_t>::max();
    auto hule = std::bind(&Kanachan::hule, std::ref(round_state), zimo_tile_, result);
    std::function<std::any()> next_step(std::move(hule));
    return next_step;
  }

  KANACHAN_THROW<std::logic_error>(_1) << action << ": An invalid action on an gang.";
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
}
#pragma GCC diagnostic pop

} // namespace Kanachan
