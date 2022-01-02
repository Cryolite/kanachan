#include "simulation/utility.hpp"

#include <boost/python/object.hpp>
#include <Python.h>
#include <random>
#include <vector>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

namespace python = boost::python;

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

bool hasattr(python::object o, char const *name)
{
  return PyObject_HasAttrString(o.ptr(), name);
}

} // namespace Kanachan
