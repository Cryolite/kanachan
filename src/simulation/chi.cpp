#include "simulation/chi.hpp"

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


namespace Kanachan{

using std::placeholders::_1;
namespace python = boost::python;

std::any chi(
  Kanachan::RoundState &round_state, std::uint_fast8_t const encode,
  python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));
  KANACHAN_ASSERT((encode < 90u));

  std::uint_fast16_t const action = round_state.onChi(encode);

  if (action <= 147u) {
    std::uint_fast8_t const tile = action / 4u;
    bool const moqi = ((action - tile * 4u) / 2u >= 2u);
    bool const lizhi = ((action - tile * 4u - moqi * 2u) == 1u);
    KANACHAN_ASSERT((!lizhi));
    auto dapai = std::bind(
      &Kanachan::dapai, std::ref(round_state), tile, moqi, lizhi, result);
    std::function<std::any()> next_step(std::move(dapai));
    return next_step;
  }

  KANACHAN_THROW<std::logic_error>(_1)
    << action << ": An invalid dapai action after chi.";
}

} // namespace Kanachan
