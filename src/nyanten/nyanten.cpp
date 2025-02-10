#include <nyanten/replacement_number.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/long.hpp>


namespace{

namespace python = boost::python;

python::long_ calculate_replacement_number(python::list tile_counts)
{
  if (python::len(tile_counts) != 34) {
    throw std::invalid_argument("`tile_count` must have 34 elements.");
  }

  std::array<std::uint_fast8_t, 34> tile_counts_{};
  for (std::size_t i = 0u; i < 34u; ++i) {
    long const tile_count = python::extract<long>(tile_counts[i]);
    if (tile_count < 0) {
      throw std::invalid_argument("Each element of `tile_count` must be non-negative.");
    }
    if (tile_count > 4) {
      throw std::invalid_argument("Each element of `tile_count` must be less than or equal to 4.");
    }
    tile_counts_[i] = static_cast<std::uint_fast8_t>(tile_count);
  }

  std::uint_fast8_t const replacement_number = Nyanten::calculateReplacementNumber(tile_counts_);
  return python::long_(replacement_number);
}

} // namespace `unnamed`

BOOST_PYTHON_MODULE(_nyanten)
{
  python::def("calculate_replacement_number", calculate_replacement_number);
}
