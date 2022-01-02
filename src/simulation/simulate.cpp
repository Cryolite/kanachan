#include "simulate.hpp"

#include "simulation/duplicated_games.hpp"
#include "simulation/game.hpp"
#include "simulation/paishan.hpp"
#include "simulation/utility.hpp"
#include "common/throw.hpp"
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/long.hpp>
#include <boost/python/object.hpp>
#include <random>
#include <algorithm>
#include <vector>
#include <array>
#include <functional>
#include <utility>
#include <stdexcept>
#include <exception>
#include <limits>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

using std::placeholders::_1;
namespace python = boost::python;

python::dict simulate(
  python::long_ simulation_mode, python::long_ baseline_grade,
  python::object baseline_model, python::long_ proposal_grade,
  python::object proposal_model, python::object external_tool)
try {
  if (simulation_mode.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`simulation_mode` is `None`.";
  }
  long simulation_mode_ = python::extract<long>(simulation_mode);

  bool const no_duplicate = simulation_mode_ & 1u;
  bool const dong_feng_zhan = simulation_mode_ & 2u;
  bool const one_versus_three = simulation_mode_ & 4u;

  std::uint_fast8_t room = std::numeric_limits<std::uint_fast8_t>::max();
  if ((simulation_mode_ & 8u) != 0u) {
    room = 0u;
  }
  if ((simulation_mode_ & 16u) != 0u) {
    if (room != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "simulation_mode: Multiple rooms specified.";
    }
    room = 1u;
  }
  if ((simulation_mode_ & 32u) != 0u) {
    if (room != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "simulation_mode: Multiple rooms specified.";
    }
    room = 2u;
  }
  if ((simulation_mode_ & 64u) != 0u) {
    if (room != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "simulation_mode: Multiple rooms specified.";
    }
    room = 3u;
  }
  if ((simulation_mode_ & 128u) != 0u) {
    if (room != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "simulation_mode: Multiple rooms specified.";
    }
    room = 4u;
  }
  if (room == std::numeric_limits<std::uint_fast8_t>::max()) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << "simulation_mode: Room not specified.";
  }

  if (baseline_grade.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`baseline_grade` is `None`.";
  }
  long const baseline_grade_ = python::extract<long>(baseline_grade);
  if (baseline_grade_ < 0 || 16 <= baseline_grade_) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << baseline_grade_ << ": An invalid baseline grade.";
  }

  if (baseline_model.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`baseline_model` is `None`.";
  }

  if (proposal_grade.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`proposal_grade` is `None`.";
  }
  long const proposal_grade_ = python::extract<long>(proposal_grade);
  if (proposal_grade_ < 0 || 16 <= proposal_grade_) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << proposal_grade_ << ": An invalid proposal grade.";
  }

  std::vector<Kanachan::Paishan> test_paishan_list;
  python::dict result;

  if (no_duplicate) {
    std::vector<std::uint_least32_t> seed = Kanachan::getRandomSeed();
    std::seed_seq ss(seed.cbegin(), seed.cend());
    std::mt19937 urng(ss);
    using Seat = std::pair<std::uint_fast8_t, python::object>;
    std::array<Seat, 4u> seats = [&]() -> std::array<Seat, 4u> {
      if (one_versus_three) {
        return std::array{
          Seat(baseline_grade_, baseline_model),
          Seat(baseline_grade_, baseline_model),
          Seat(baseline_grade_, baseline_model),
          Seat(proposal_grade_, proposal_model)
        };
      }
      return std::array{
        Seat(baseline_grade_, baseline_model),
        Seat(baseline_grade_, baseline_model),
        Seat(proposal_grade_, proposal_model),
        Seat(proposal_grade_, proposal_model)
      };
    }();
    std::shuffle(seats.begin(), seats.end(), urng);
    Kanachan::simulateGame(
      urng, room, dong_feng_zhan, seats, external_tool, test_paishan_list,
      result);
    return result;
  }

  Kanachan::simulateDuplicatedGames(
    room, dong_feng_zhan, one_versus_three, baseline_grade_, baseline_model,
    proposal_grade_, proposal_model, external_tool, test_paishan_list, result);
  return result;
}
catch (...) {
  std::terminate();
}

python::dict test(
  python::long_ const simulation_mode, python::tuple const grades,
  python::object test_model, python::object external_tool,
  python::list const test_paishan_list)
try {
  if (simulation_mode.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`simulation_mode` is `None`.";
  }
  long simulation_mode_ = python::extract<long>(simulation_mode);

  bool const no_duplicate = simulation_mode_ & 1u;
  bool const dong_feng_zhan = simulation_mode_ & 2u;
  bool const one_versus_three = simulation_mode_ & 4u;

  std::uint_fast8_t room = std::numeric_limits<std::uint_fast8_t>::max();
  if ((simulation_mode_ & 8u) != 0u) {
    room = 0u;
  }
  if ((simulation_mode_ & 16u) != 0u) {
    if (room != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "simulation_mode: Multiple rooms specified.";
    }
    room = 1u;
  }
  if ((simulation_mode_ & 32u) != 0u) {
    if (room != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "simulation_mode: Multiple rooms specified.";
    }
    room = 2u;
  }
  if ((simulation_mode_ & 64u) != 0u) {
    if (room != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "simulation_mode: Multiple rooms specified.";
    }
    room = 3u;
  }
  if ((simulation_mode_ & 128u) != 0u) {
    if (room != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "simulation_mode: Multiple rooms specified.";
    }
    room = 4u;
  }
  if (room == std::numeric_limits<std::uint_fast8_t>::max()) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << "simulation_mode: Room not specified.";
  }

  if (grades.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`grades` is `None`.";
  }
  if (python::len(grades) != 4) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << python::len(grades) << ": An invalid length.";
  }

  if (test_model.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`grades` is `None`.";
  }

  if (external_tool.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`external_tool` is `None`.";
  }

  if (test_paishan_list.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`test_paishan_list` is `None`.";
  }
  if (python::len(test_paishan_list) == 0) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`test_paishan_list` is empty.";
  }

  std::vector<Kanachan::Paishan> test_paishan_list_;
  for (std::size_t i = 0; i < python::len(test_paishan_list); ++i) {
    python::list paishan = [test_paishan_list, i]() -> python::list {
      python::object o = test_paishan_list[i];
      python::extract<python::list> paishan_(o);
      if (!paishan_.check()) {
        KANACHAN_THROW<std::invalid_argument>("test_paishan_list: A type error.");
      }
      return paishan_();
    }();
    if (python::len(paishan) != 136u) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "test_paishan_list: An wrong length (" << python::len(paishan) << ").";
    }
    Kanachan::Paishan paishan_(paishan);
    test_paishan_list_.push_back(std::move(paishan_));
  }

  python::dict result;

  std::vector<std::uint_least32_t> seed = Kanachan::getRandomSeed();
  std::seed_seq ss(seed.cbegin(), seed.cend());
  std::mt19937 urng(ss);
  using Seat = std::pair<std::uint_fast8_t, python::object>;
  std::array<Seat, 4u> seats = [&]() -> std::array<Seat, 4u> {
    python::extract<python::long_> grade0(grades[0]);
    python::extract<long> grade0_(grade0());
    python::extract<python::long_> grade1(grades[1]);
    python::extract<long> grade1_(grade1());
    python::extract<python::long_> grade2(grades[2]);
    python::extract<long> grade2_(grade2());
    python::extract<python::long_> grade3(grades[3]);
    python::extract<long> grade3_(grade3());
    return std::array{
      Seat(grade0_(), test_model),
      Seat(grade1_(), test_model),
      Seat(grade2_(), test_model),
      Seat(grade3_(), test_model)
    };
  }();
  Kanachan::simulateGame(
    urng, room, dong_feng_zhan, seats, external_tool, test_paishan_list_,
    result);
  return result;
}
catch (...) {
  std::terminate();
}

} // namespace Kanachan
