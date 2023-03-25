#include <simulation/model_wrapper.hpp>

#include <common/assert.hpp>
#include <common/throw.hpp>
#include <boost/python/import.hpp>
#include <boost/python/args.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/object.hpp>
#include <boost/python/handle.hpp>
#include <boost/python/errors.hpp>
#include <Python.h>
#include <string>
#include <functional>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <cstdint>


namespace Kanachan{

namespace{

using std::placeholders::_1;
namespace python = boost::python;

} // namespace *unnamed*

ModelWrapper::ModelWrapper(
  std::string const &device, python::object dtype, python::object model)
  : device_(device), dtype_(dtype), model_(model)
{
  KANACHAN_ASSERT((!dtype_.is_none()));
  KANACHAN_ASSERT((!model_.is_none()));
}

void ModelWrapper::swap(ModelWrapper &rhs) noexcept
{
  using std::swap;
  swap(device_, rhs.device_);
  swap(dtype_,  rhs.dtype_);
  swap(model_,  rhs.model_);
}

void swap(ModelWrapper &lhs, ModelWrapper &rhs) noexcept
{
  lhs.swap(rhs);
}

ModelWrapper &ModelWrapper::operator=(ModelWrapper const &rhs)
{
  ModelWrapper(rhs).swap(*this);
  return *this;
}

ModelWrapper &ModelWrapper::operator=(ModelWrapper &&rhs) noexcept
{
  ModelWrapper(std::move(rhs)).swap(*this);
  return *this;
}

std::uint_fast16_t ModelWrapper::operator()(python::object features) const
{
  KANACHAN_ASSERT((!features.is_none()));
  KANACHAN_ASSERT((python::len(features) == 4));

  python::object torch = python::import("torch");
  python::object constants = python::import("kanachan.training.constants");

  python::object candidates;
  python::object prediction;
  try {
    python::object tensor = torch.attr("tensor");
    python::dict kwargs;

    python::object sparse = features[0];
    while (python::len(sparse) < constants.attr("MAX_NUM_ACTIVE_SPARSE_FEATURES")) {
      // Padding.
      sparse.attr("append")(
        python::getattr(constants, "NUM_TYPES_OF_SPARSE_FEATURES"));
    }
    kwargs["device"] = device_;
    kwargs["dtype"] = torch.attr("int");
    sparse = tensor(*python::make_tuple(sparse), **kwargs);
    sparse = torch.attr("unsqueeze")(sparse, 0);

    python::object numeric = features[1];
    for (long i = 2; i < constants.attr("NUM_NUMERIC_FEATURES"); ++i) {
      // Scaling.
      numeric[i] = double(python::extract<long>(numeric[i])()) / 10000.0;
    }
    kwargs["device"] = device_;
    kwargs["dtype"] = dtype_;
    numeric = tensor(*python::make_tuple(numeric), **kwargs);
    numeric = torch.attr("unsqueeze")(numeric, 0);

    // `progression` must be deep-copied since `features[2]` refers to the
    // `progression_` data member of the `RoundState` class.
    python::object progression = python::list(features[2]);
    while (python::len(progression) < constants.attr("MAX_LENGTH_OF_PROGRESSION_FEATURES")) {
      // Padding.
      progression.attr("append")(
        python::getattr(constants, "NUM_TYPES_OF_PROGRESSION_FEATURES"));
    }
    kwargs["device"] = device_;
    kwargs["dtype"] = torch.attr("int");
    progression = tensor(*python::make_tuple(progression), **kwargs);
    progression = torch.attr("unsqueeze")(progression, 0);

    candidates = features[3];
    // Append `<V>`.
    candidates.attr("append")(python::getattr(constants, "NUM_TYPES_OF_ACTIONS"));
    while (python::len(candidates) < constants.attr("MAX_NUM_ACTION_CANDIDATES")) {
      // Padding.
      candidates.attr("append")(python::getattr(constants, "NUM_TYPES_OF_ACTIONS") + 1);
    }
    kwargs["device"] = device_;
    kwargs["dtype"] = torch.attr("int");
    candidates = tensor(*python::make_tuple(candidates), **kwargs);
    candidates = torch.attr("unsqueeze")(candidates, 0);

    prediction = model_(sparse, numeric, progression, candidates);
    if (prediction.attr("dim")() != 2) {
      long const dim = python::extract<long>(prediction.attr("dim")())();
      KANACHAN_THROW<std::runtime_error>(_1)
        << dim << ": An invalid dimension for `prediction`.";
    }
    if (prediction.attr("size")(0) != candidates.attr("size")(0)) {
      long const size = python::extract<long>(prediction.attr("size")(0))();
      KANACHAN_THROW<std::runtime_error>(_1)
        << size << ": An invalid size for the 0th dimension of `prediction`.";
    }
    if (prediction.attr("size")(1) != constants.attr("MAX_NUM_ACTION_CANDIDATES")) {
      long const size = python::extract<long>(prediction.attr("size")(1))();
      KANACHAN_THROW<std::runtime_error>(_1)
        << size << ": An invalid size for the 1st dimension of `prediction`.";
    }
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

  python::object mask = torch.attr("lt")(
    candidates, python::getattr(constants, "NUM_TYPES_OF_ACTIONS"));
  candidates = torch.attr("squeeze")(candidates);
  prediction = prediction[mask];
  prediction = torch.attr("argmax")(prediction);
  prediction = prediction.attr("item")();
  python::extract<long> prediction_(prediction);
  if (!prediction_.check()) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << prediction.attr("__class__").attr("__name__")
      << ": An invalid type of `prediction`.";
  }
  if (prediction >= candidates.attr("size")(0)) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << prediction_() << ": `prediction` is out-of-range.";
  }

  python::object action = candidates[prediction];
  action = action.attr("item")();
  python::extract<long> action_(action);
  if (!action_.check()) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << prediction.attr("__class__").attr("__name__")
      << ": An invalid type of `action`.";
  }
  if (action_() >= constants.attr("NUM_TYPES_OF_ACTIONS")) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << action_() << ": `action` is out-of-range.";
  }
  return action_();
}

} // namespace Kanachan
