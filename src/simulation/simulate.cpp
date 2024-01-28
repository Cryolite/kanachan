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
#include <iostream>
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
  python::object device, python::object dtype, long const room, long const baseline_grade,
  python::object baseline_model, python::list baseline_keys_to_be_deleted,
  long const proposed_grade, python::object proposed_model,
  python::list proposed_keys_to_be_deleted, long const simulation_mode,
  long const num_simulation_sets, long const batch_size, long const concurrency,
  python::object progress)
try {
  if (PyGILState_Check() == 0) {
    KANACHAN_THROW<std::runtime_error>("GIL must be held.");
  }
  if (device.is_none()) {
    KANACHAN_THROW<std::invalid_argument>("`device` must not be `None`.");
  }
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
    device, dtype, room, baseline_grade, baseline_model, baseline_keys_to_be_deleted,
    proposed_grade, proposed_model, proposed_keys_to_be_deleted, simulation_mode,
    num_simulation_sets, batch_size, concurrency, progress);
  python::list result = simulator.run();
  return result;
}
catch (std::exception const &e) {
  std::cerr << e.what() << std::endl;
  std::abort();
}
catch (python::error_already_set const &) {
  try {
    Kanachan::translatePythonException();
  }
  catch (std::runtime_error const &e) {
    std::cerr << e.what() << std::endl;
    std::abort();
  }
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
    python::import("torch").attr("device")("cpu"), python::import("torch").attr("float64"),
    test_model, python::list(), test_model, python::list(), 1u);

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

  std::mt19937 urng;
  Kanachan::Deciders deciders{
    std::bind_front(&Kanachan::DecisionMaker::operator(), p_test_decision_maker, false),
    std::bind_front(&Kanachan::DecisionMaker::operator(), p_test_decision_maker, false),
    std::bind_front(&Kanachan::DecisionMaker::operator(), p_test_decision_maker, false),
    std::bind_front(&Kanachan::DecisionMaker::operator(), p_test_decision_maker, false)
  };
  std::array<std::uint_fast8_t, 4u> grades_{
    static_cast<std::uint_fast8_t>(python::extract<long>(grades[0])()),
    static_cast<std::uint_fast8_t>(python::extract<long>(grades[1])()),
    static_cast<std::uint_fast8_t>(python::extract<long>(grades[2])()),
    static_cast<std::uint_fast8_t>(python::extract<long>(grades[2])())
  };
  return Kanachan::simulateGame(
    urng, room, dong_feng_zhan, deciders, grades_, test_paishan_list_, std::stop_token());
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
