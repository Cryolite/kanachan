#include "simulation/peng.hpp"

#include "simulation/dapai.hpp"
#include "simulation/round_state.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <functional>
#include <any>
#include <utility>
#include <stdexcept>
#include <cstdint>


namespace{

using std::placeholders::_1;
namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

std::any peng(
  Kanachan::RoundState &round_state, std::uint_fast8_t const encode, python::dict result)
{
  if (encode >= 40u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(encode);
  }
  if (result.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`result` must not be a `None`.");
  }

  std::uint_fast16_t const action = round_state.onPeng(encode);

  if (action >= 148u) {
    KANACHAN_THROW<std::runtime_error>(_1) << action << ": An invalid dapai action after peng.";
  }

  std::uint_fast8_t const tile = action / 4u;
  bool const moqi = ((action - tile * 4u) / 2u >= 2u);
  KANACHAN_ASSERT((!moqi));
  bool const lizhi = ((action - tile * 4u - moqi * 2u) == 1u);
  KANACHAN_ASSERT((!lizhi));
  auto dapai = std::bind(&Kanachan::dapai, std::ref(round_state), tile, moqi, lizhi, result);
  std::function<std::any()> next_step(std::move(dapai));
  return next_step;
}

} // namespace Kanachan
