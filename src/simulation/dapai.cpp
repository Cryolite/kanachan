#include "simulation/dapai.hpp"

#include "simulation/sijia_lizhi.hpp"
#include "simulation/sigang_sanle.hpp"
#include "simulation/sifeng_lianda.hpp"
#include "simulation/huangpai_pingju.hpp"
#include "simulation/hule.hpp"
#include "simulation/daminggang.hpp"
#include "simulation/peng.hpp"
#include "simulation/chi.hpp"
#include "simulation/zimo.hpp"
#include "simulation/round_state.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/dict.hpp>
#include <functional>
#include <any>
#include <utility>
#include <stdexcept>
#include <limits>
#include <cstdint>


namespace Kanachan{

using std::placeholders::_1;
namespace python = boost::python;

std::any dapai(
  Kanachan::RoundState &round_state, std::uint_fast8_t const tile,
  bool const moqi, bool const lizhi, python::dict result)
{
  auto const [dapai_seat, action] = round_state.onDapai(tile, moqi, lizhi);

  if (action == 221u) {
    // Skip
    KANACHAN_ASSERT(
      (dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()));
    if (round_state.getNumLeftTiles() == 0u) {
      auto huangpai_pingju = std::bind(
        &Kanachan::huangpaiPingju, std::ref(round_state), result);
      std::function<std::any()> next_step(std::move(huangpai_pingju));
      return next_step;
    }
    auto zimo = std::bind(&Kanachan::zimo, std::ref(round_state), result);
    std::function<std::any()> next_step(std::move(zimo));
    return next_step;
  }

  if (dapai_seat != static_cast<std::uint_fast8_t>(-1) && action != static_cast<uint_fast16_t>(-1)) {
    // Chi, Peng, or Da Ming Gang (チー・ポン・大明槓)
    KANACHAN_ASSERT((dapai_seat < 4u));
    if (222u <= action && action <= 311u) {
      // Chi (チー)
      std::uint_fast8_t const encode = action - 222u;
      auto chi = std::bind(
        &Kanachan::chi, std::ref(round_state), encode, result);
      std::function<std::any()> next_step(std::move(chi));
      return next_step;
    }
    if (312u <= action && action <= 431u) {
      // Peng (ポン)
      std::uint_fast8_t const relseat = (action - 312u) / 40u;
      std::uint_fast8_t const encode = action - 312u - relseat * 40u;
      auto peng = std::bind(
        &Kanachan::peng, std::ref(round_state), encode, result);
      std::function<std::any()> next_step(std::move(peng));
      return next_step;
    }
    if (432u <= action && action <= 542u) {
      // Da Ming Gang (大明槓)
      auto daminggang = std::bind(
        &Kanachan::daminggang, std::ref(round_state), result);
      std::function<std::any()> next_step(std::move(daminggang));
      return next_step;
    }
    KANACHAN_THROW<std::logic_error>(_1) << action << ": An invalid action.";
  }

  if (action == 543u) {
    // Rong (栄和)
    KANACHAN_ASSERT(
      (dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()));
    std::uint_fast8_t const zimo_tile = std::numeric_limits<std::uint_fast8_t>::max();
    auto hule = std::bind(
      &Kanachan::hule, std::ref(round_state), zimo_tile, result);
    std::function<std::any()> next_step(std::move(hule));
    return next_step;
  }

  if (action == std::numeric_limits<std::uint_fast16_t>::max()) {
    // Liu Ju (流局) in the middle.
    if (round_state.checkSifengLianda()) {
      // Si Feng Lian Da (四風連打)
      auto sifeng_lianda = std::bind(
        &Kanachan::sifengLianda, std::ref(round_state), result);
      std::function<std::any()> next_step(std::move(sifeng_lianda));
      return next_step;
    }
    if (round_state.checkSigangSanle()) {
      // Si Gang San Le (四槓散了)
      auto sigang_sanle = std::bind(
        &Kanachan::sigangSanle, std::ref(round_state), result);
      std::function<std::any()> next_step(std::move(sigang_sanle));
      return next_step;
    }
    if (round_state.checkSijiaLizhi()) {
      // Si Jia Li Zhi (四家立直)
      auto sijia_lizhi = std::bind(
        &Kanachan::sijiaLizhi, std::ref(round_state), result);
      std::function<std::any()> next_step(std::move(sijia_lizhi));
      return next_step;
    }
  }

  KANACHAN_THROW<std::logic_error>(_1) << action << ": An invalid action.";
}

} // namespace Kanachan
