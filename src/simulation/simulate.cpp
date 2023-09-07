#include "simulate.hpp"

#include "simulation/game.hpp"
#include "simulation/paishan.hpp"
#include "simulation/simulator.hpp"
#include "simulation/decision_maker.hpp"
#include "simulation/utility.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/import.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/str.hpp>
#include <boost/python/long.hpp>
#include <boost/python/object.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/errors.hpp>
#include <Python.h>
#include <random>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <tuple>
#include <utility>
#include <memory>
#include <stdexcept>
#include <exception>
#include <cstdint>
#include <limits>
#include <cstddef>


namespace{

using std::placeholders::_1;
namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

python::list simulate(
  std::string const &device, python::object dtype, long const room,
  long const baseline_grade, python::object baseline_model,
  long const proposed_grade, python::object proposed_model,
  long const simulation_mode, long const num_simulation_sets,
  long const batch_size, long const concurrency)
try {
  if (dtype.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`dtype` must not be `None`.");
  }
  if (room < 0 || 4 < room) {
    KANACHAN_THROW<std::invalid_argument>(_1) << room << ": An invalid value for `room`.";
  }
  if (baseline_grade < 0 || 16 <= baseline_grade) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << baseline_grade << ": An invalid value for `baseline_grade`.";
  }
  if (baseline_model.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`baseline_model` must not be `None`.");
  }
  if (proposed_grade < 0 || 16 <= proposed_grade) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << proposed_grade << ": An invalid value for `proposed_grade`.";
  }
  if (proposed_model.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`proposed_model` must not be `None`.");
  }

  Kanachan::Simulator simulator(
    device, dtype, room, baseline_grade, baseline_model, proposed_grade, proposed_model,
    simulation_mode, num_simulation_sets, batch_size, concurrency);
  return simulator.run();
}
catch (std::exception const &) {
  throw;
}
catch (python::error_already_set const &) {
  Kanachan::translatePythonException();
}
catch (...) {
  std::terminate();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
}
#pragma GCC diagnostic pop

std::shared_ptr<Kanachan::GameLog> test(
  python::long_ const simulation_mode, python::tuple const grades,
  python::object test_model, python::list const test_paishan_list)
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
  auto p_test_decision_maker = std::make_shared<Kanachan::DecisionMaker>(
    "cpu", python::import("torch").attr("float64"), test_model, 1u, false);

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
  using Seat = std::pair<std::uint_fast8_t, std::shared_ptr<Kanachan::DecisionMaker>>;
  std::array<Seat, 4u> seats = [&]() -> std::array<Seat, 4u>
  {
    python::extract<python::long_> grade0(grades[0]);
    python::extract<long> grade0_(grade0());
    python::extract<python::long_> grade1(grades[1]);
    python::extract<long> grade1_(grade1());
    python::extract<python::long_> grade2(grades[2]);
    python::extract<long> grade2_(grade2());
    python::extract<python::long_> grade3(grades[3]);
    python::extract<long> grade3_(grade3());
    return std::array{
      Seat(grade0_(), p_test_decision_maker),
      Seat(grade1_(), p_test_decision_maker),
      Seat(grade2_(), p_test_decision_maker),
      Seat(grade3_(), p_test_decision_maker)
    };
  }();
  return Kanachan::simulateGame(
    seed, room, dong_feng_zhan, seats, test_paishan_list_, std::stop_token());
}
catch (std::exception const &) {
  throw;
}
catch (python::error_already_set const &) {
  Kanachan::translatePythonException();
}
catch (...) {
  std::terminate();
}

} // namespace Kanachan
