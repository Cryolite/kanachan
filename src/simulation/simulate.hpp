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

void simulate(
  std::string const &device, boost::python::object dtype,
  boost::python::long_ simulation_mode, boost::python::long_ baseline_grade,
  boost::python::object baseline_model, boost::python::long_ proposed_grade,
  boost::python::object proposed_model, boost::python::dict result);

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
