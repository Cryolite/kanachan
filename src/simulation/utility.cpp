#define PY_SSIZE_T_CLEAN
#include "simulation/utility.hpp"

#include "simulation/gil.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/import.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/str.hpp>
#include <boost/python/object.hpp>
#include <Python.h>
#include <random>
#include <vector>
#include <string>
#include <stdexcept>
#include <exception>
#include <cstdint>
#include <cstddef>


namespace{

namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

std::vector<std::uint_least32_t> getRandomSeed()
{
  constexpr std::size_t state_size = std::mt19937::state_size;

  std::random_device rand;
  std::vector<std::uint_least32_t> seed;
  seed.reserve(state_size);
  for (std::size_t i = 0; i < state_size; ++i) {
    std::uint_least32_t const seed_ = rand();
    seed.push_back(seed_);
  }
  return seed;
}

void translatePythonException()
{
    std::exception_ptr const p = std::current_exception();
    if (!p) {
        KANACHAN_THROW<std::runtime_error>("No exception.");
    }

    try {
        std::rethrow_exception(p);
    }
    catch (python::error_already_set const &) {
        Kanachan::GIL::RecursiveLock gil_lock;

        auto const [type, value, traceback] = [](){
            PyObject *p_type = nullptr;
            PyObject *p_value = nullptr;
            PyObject *p_traceback = nullptr;
            PyErr_Fetch(&p_type, &p_value, &p_traceback);

            python::object type{ python::handle<>(p_type) };

            python::object value;
            if (p_value != nullptr) {
                value = python::object{ python::handle<>(p_value) };
            }
            else {
                value = python::object();
            }

            python::object traceback;
            if (p_traceback != nullptr) {
                traceback = python::object{python::handle<>(p_traceback)};
            }

            return std::tuple(type, value, traceback);
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
}

} // namespace Kanachan
