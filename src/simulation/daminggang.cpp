#include "simulation/daminggang.hpp"

#include "simulation/zimo.hpp"
#include "simulation/round_state.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <any>
#include <utility>
#include <stdexcept>


namespace {

namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

std::any daminggang(Kanachan::RoundState &round_state, python::dict result)
{
  if (result.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`result` must not be a `None`.");
  }

  round_state.onDaminggang();

  // Si Gang San Le (四槓散了) の成立は打牌直後．

  // Zimo (自摸)
  auto zimo = std::bind(&Kanachan::zimo, std::ref(round_state), result);
  std::function<std::any()> next_step(std::move(zimo));
  return next_step;
}

} // namespace Kanachan
