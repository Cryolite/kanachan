#if !defined(KANACHAN_SIMULATION_SIMULATE_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_SIMULATE_HPP_INCLUDE_GUARD

#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/long.hpp>
#include <boost/python/object.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <string>


namespace Kanachan{

boost::python::list simulate(
  std::string const &device, boost::python::object dtype,
  long baseline_grade, boost::python::object baseline_model,
  long proposed_grade, boost::python::object proposed_model,
  long simulation_mode, long num_simulation_sets,
  long batch_size, long concurrency);

boost::python::dict test(
  boost::python::long_ simulation_mode, boost::python::tuple grades,
  boost::python::object test_model, boost::python::list test_paishan_list);

} // namespace Kanachan


BOOST_PYTHON_MODULE(_simulation)
{
  boost::python::def("simulate", &Kanachan::simulate);
  boost::python::def("test", &Kanachan::test);
} // BOOST_PYTHON_MODULE(_simulation)


#endif // !defined(KANACHAN_SIMULATION_SIMULATE_HPP_INCLUDE_GUARD)
