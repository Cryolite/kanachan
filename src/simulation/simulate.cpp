#include "simulate.hpp"

#include "simulation/duplicated_games.hpp"
#include "simulation/game.hpp"
#include "simulation/paishan.hpp"
#include "simulation/model_wrapper.hpp"
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
#include <stdexcept>
#include <exception>
#include <cstdint>
#include <limits>
#include <cstddef>


namespace Kanachan{

using std::placeholders::_1;
namespace python = boost::python;

python::list simulate(
  std::string const &device, python::object dtype,
  python::long_ simulation_mode, python::long_ baseline_grade,
  python::object baseline_model, python::long_ proposed_grade,
  python::object proposed_model)
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
  Kanachan::ModelWrapper baseline_model_wrapper(device, dtype, baseline_model);

  if (proposed_grade.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`proposed_grade` is `None`.";
  }
  long const proposed_grade_ = python::extract<long>(proposed_grade);
  if (proposed_grade_ < 0 || 16 <= proposed_grade_) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << proposed_grade_ << ": An invalid value for `proposed_grade`.";
  }

  if (proposed_model.is_none()) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "`proposed_model` is `None`.";
  }
  Kanachan::ModelWrapper proposed_model_wrapper(device, dtype, proposed_model);

  std::vector<Kanachan::Paishan> test_paishan_list;

  if (no_duplicate) {
    std::vector<std::uint_least32_t> seed = Kanachan::getRandomSeed();
    std::seed_seq ss(seed.cbegin(), seed.cend());
    std::mt19937 urng(ss);

    std::array<bool, 4u> flags = {false, false, !one_versus_three, true};
    std::shuffle(flags.begin(), flags.end(), urng);

    using Seat = std::pair<std::uint_fast8_t, Kanachan::ModelWrapper>;
    std::array<Seat, 4u> seats = {
      flags[0u] ? Seat(proposed_grade_, proposed_model_wrapper) : Seat(baseline_grade_, baseline_model_wrapper),
      flags[1u] ? Seat(proposed_grade_, proposed_model_wrapper) : Seat(baseline_grade_, baseline_model_wrapper),
      flags[2u] ? Seat(proposed_grade_, proposed_model_wrapper) : Seat(baseline_grade_, baseline_model_wrapper),
      flags[3u] ? Seat(proposed_grade_, proposed_model_wrapper) : Seat(baseline_grade_, baseline_model_wrapper)
    };

    python::dict result = Kanachan::simulateGame(
      urng, room, dong_feng_zhan, seats, test_paishan_list);

    result["proposed"] = python::list();
    result["proposed"].attr("append")(flags[0u] ? 1 : 0);
    result["proposed"].attr("append")(flags[1u] ? 1 : 0);
    result["proposed"].attr("append")(flags[2u] ? 1 : 0);
    result["proposed"].attr("append")(flags[3u] ? 1 : 0);

    python::list results;
    results.attr("append")(result);
    return results;
  }

  return Kanachan::simulateDuplicatedGames(
    room, dong_feng_zhan, one_versus_three, baseline_grade_,
    baseline_model_wrapper, proposed_grade_, proposed_model_wrapper,
    test_paishan_list);
}
catch (std::exception const &) {
  throw;
}
catch (python::error_already_set const &) {
  auto const [type, value, traceback] = [](){
    PyObject *p_type = nullptr;
    PyObject *p_value = nullptr;
    PyObject *p_traceback = nullptr;
    PyErr_Fetch(&p_type, &p_value, &p_traceback);
    python::object type_{python::handle<>(p_type)};
    python::object value_;
    if (p_value != nullptr) {
      value_ = python::object{python::handle<>(p_value)};
    }
    else {
      value_ = python::object();
    }
    python::object traceback_;
    if (p_traceback != nullptr) {
      traceback_ = python::object{python::handle<>(p_traceback)};
    }
    else {
      traceback_ = python::object();
    }
    return std::tuple(type_, value_, traceback_);
  }();

  python::object m = python::import("traceback");

  if (python::extract<std::string>(value).check()) {
    python::object o = m.attr("format_tb")(traceback);
    o = python::str("").attr("join")(o);
    std::string message = python::extract<std::string>(o);
    message += python::extract<std::string>(value);
    KANACHAN_THROW<std::runtime_error>(message);
  }

  python::object o = m.attr("format_exception")(type, value, traceback);
  o = python::str("").attr("join")(o);
  python::extract<std::string> str(o);
  KANACHAN_ASSERT((str.check()));
  KANACHAN_THROW<std::runtime_error>(str());
}
catch (...) {
  std::terminate();
}

python::dict test(
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
  Kanachan::ModelWrapper test_model_wrapper(
    "cpu", python::import("torch").attr("float64"), test_model);

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
  using Seat = std::pair<std::uint_fast8_t, Kanachan::ModelWrapper>;
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
      Seat(grade0_(), test_model_wrapper),
      Seat(grade1_(), test_model_wrapper),
      Seat(grade2_(), test_model_wrapper),
      Seat(grade3_(), test_model_wrapper)
    };
  }();
  return Kanachan::simulateGame(
    urng, room, dong_feng_zhan, seats, test_paishan_list_);
}
catch (std::exception const &) {
  throw;
}
catch (python::error_already_set const &) {
  auto const [type, value, traceback] = [](){
    PyObject *p_type = nullptr;
    PyObject *p_value = nullptr;
    PyObject *p_traceback = nullptr;
    PyErr_Fetch(&p_type, &p_value, &p_traceback);
    python::object type_{python::handle<>(p_type)};
    python::object value_;
    if (p_value != nullptr) {
      value_ = python::object{python::handle<>(p_value)};
    }
    else {
      value_ = python::object();
    }
    python::object traceback_;
    if (p_traceback != nullptr) {
      traceback_ = python::object{python::handle<>(p_traceback)};
    }
    else {
      traceback_ = python::object();
    }
    return std::tuple(type_, value_, traceback_);
  }();

  python::object m = python::import("traceback");

  if (python::extract<std::string>(value).check()) {
    python::object o = m.attr("format_tb")(traceback);
    o = python::str("").attr("join")(o);
    std::string message = python::extract<std::string>(o);
    message += python::extract<std::string>(value);
    KANACHAN_THROW<std::runtime_error>(message);
  }

  python::object o = m.attr("format_exception")(type, value, traceback);
  o = python::str("").attr("join")(o);
  python::extract<std::string> str(o);
  KANACHAN_ASSERT((str.check()));
  KANACHAN_THROW<std::runtime_error>(str());
}
catch (...) {
  std::terminate();
}

} // namespace Kanachan
