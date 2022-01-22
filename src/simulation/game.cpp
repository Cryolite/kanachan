#include "simulation/game.hpp"

#include "simulation/round.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include "simulation/model_wrapper.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <random>
#include <vector>
#include <array>
#include <functional>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

using std::placeholders::_1;
namespace python = boost::python;

python::dict simulateGame(
  std::mt19937 &urng, std::uint_fast8_t room, bool dong_feng_zhan,
  std::array<std::pair<std::uint_fast8_t, Kanachan::ModelWrapper>, 4u> const &seats,
  std::vector<Kanachan::Paishan> const &test_paishan_list)
{
  KANACHAN_ASSERT((room < 5u));
  for (auto [grade, model] : seats) {
    KANACHAN_ASSERT((grade < 16u));
    KANACHAN_ASSERT((!model.is_none()));
  }

  bool const test = !test_paishan_list.empty();
  std::size_t i = 0u;

  Kanachan::GameState game_state(room, dong_feng_zhan, seats);
  python::dict result;
  bool end_of_game = false;
  while (!end_of_game) {
    if (test) {
      if (i >= test_paishan_list.size()) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << "The number of test PaiShan is too small: i == " << i
          << ", test_paishan_list.size() == " << test_paishan_list.size();
      }
      Kanachan::Paishan const &test_paishan = test_paishan_list[i++];
      end_of_game = Kanachan::simulateRound(
        urng, game_state, &test_paishan, result);
      if (end_of_game && i != test_paishan_list.size()) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << "The number of test pai shan is too large: i == " << i
          << ", test_paishan_list.size() == " << test_paishan_list.size();
      }
    }
    else {
      end_of_game = Kanachan::simulateRound(urng, game_state, nullptr, result);
    }
  }

  result["final_ranking"] = python::list();
  result["final_scores"] = python::list();
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    std::uint_fast8_t const final_ranking = game_state.getPlayerRanking(i);
    std::int_fast32_t const final_score = game_state.getPlayerScore(i);
    python::extract<python::list>(result["final_ranking"])().append(final_ranking);
    python::extract<python::list>(result["final_scores"])().append(final_score);
  }

  return result;
}

} // namespace Kanachan
