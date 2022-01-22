#include "simulation/zimo.hpp"

#include "simulation/jiuzhong_jiupai.hpp"
#include "simulation/hule.hpp"
#include "simulation/jiagang.hpp"
#include "simulation/angang.hpp"
#include "simulation/dapai.hpp"
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

std::any zimo(Kanachan::RoundState &round_state, python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));

  auto const [action, zimo_tile] = round_state.onZimo();

  if (action <= 147u) {
    // Da Pai (打牌)
    // この時点ではロンされる可能性があるため立直は成立しない．
    // また，立直が確定しない限り四家立直も確定しない．
    KANACHAN_ASSERT((zimo_tile < 37u));
    KANACHAN_ASSERT((action / 4u == zimo_tile));
    // 親の配牌14枚のうちどれが第一自摸であるかの区別が存在しないため，
    // 親の第一打牌は常に手出しになる．
    bool const moqi = round_state.getNumLeftTiles() == 69u ? false : ((action - zimo_tile * 4u) / 2u == 1u);
    bool const lizhi = ((action - zimo_tile * 4u - moqi * 2u) == 1u);
    auto dapai = std::bind(
      &Kanachan::dapai, std::ref(round_state), zimo_tile, moqi, lizhi, result);
    std::function<std::any()> next_step(std::move(dapai));
    return next_step;
  }

  if (action == std::numeric_limits<std::uint_fast16_t>::max()) {
    // Da Pai after Li Zhi (立直後の打牌)
    KANACHAN_ASSERT((zimo_tile < 37u));
    auto dapai = std::bind(
      &Kanachan::dapai, std::ref(round_state), zimo_tile, /*moqi = */true,
      /*lizhi = */false, result);
    std::function<std::any()> next_step(std::move(dapai));
    return next_step;
  }

  if (/*148u <= action && */action <= 181u) {
    // An Gang (暗槓)
    // 暗槓は槍槓が成立する可能性があるため，この時点では四槓散了は確定しない．
    KANACHAN_ASSERT((zimo_tile < 37u));
    std::uint_fast8_t const encode = action - 148u;
    auto angang = std::bind(
      &Kanachan::angang, std::ref(round_state), zimo_tile, encode, result);
    std::function<std::any()> next_step(std::move(angang));
    return next_step;
  }

  if (/*182u <= action && */action <= 218u) {
    // Jia Gang (加槓)
    // 加槓は槍槓が成立する可能性があるため，この時点では四槓散了は確定しない．
    KANACHAN_ASSERT((zimo_tile < 37u));
    std::uint_fast8_t const encode = action - 182u;
    auto jiagang = std::bind(
      &Kanachan::jiagang, std::ref(round_state), zimo_tile, encode, result);
    std::function<std::any()> next_step(std::move(jiagang));
    return next_step;
  }

  if (action == 219u) {
    // Zi Mo Hu (自摸和)
    KANACHAN_ASSERT((zimo_tile < 37u));
    auto hule = std::bind(&Kanachan::hule, std::ref(round_state), zimo_tile, result);
    std::function<std::any()> next_step(std::move(hule));
    return next_step;
  }

  if (action == 220u) {
    // Jiu Zhong Jiu Pai (九種九牌)
    KANACHAN_ASSERT((zimo_tile == std::numeric_limits<std::uint_fast8_t>::max()));
    auto jiuzhong_jiupai = std::bind(
      &Kanachan::jiuzhongJiupai, std::ref(round_state), result);
    std::function<std::any()> next_step(std::move(jiuzhong_jiupai));
    return next_step;
  }

  KANACHAN_THROW<std::runtime_error>(_1)
    << action << ": An invalid action on zimo.";
}

} // namespace Kanachan
