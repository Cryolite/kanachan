#include "simulation/daminggang.hpp"

#include "simulation/zimo.hpp"
#include "simulation/round_state.hpp"
#include "common/assert.hpp"
#include <boost/python/dict.hpp>
#include <functional>
#include <any>
#include <utility>


namespace Kanachan{

using std::placeholders::_1;
namespace python = boost::python;

std::any daminggang(Kanachan::RoundState &round_state, python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));

  round_state.onDaminggang();

  // Si Gang San Le (四槓散了) の成立は打牌直後．

  // Zimo (自摸)
  auto zimo = std::bind(&Kanachan::zimo, std::ref(round_state), result);
  std::function<std::any()> next_step(std::move(zimo));
  return next_step;
}

} // namespace Kanachan
