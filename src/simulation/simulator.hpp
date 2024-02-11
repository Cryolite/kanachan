#if !defined(KANACHAN_SIMULATION_SIMULATOR_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_SIMULATOR_HPP_INCLUDE_GUARD

#include <boost/python/list.hpp>
#include <boost/python/str.hpp>
#include <boost/python/object.hpp>
#include <string>
#include <memory>
#include <cstdint>
#include <cstddef>


namespace Kanachan {

class Simulator
{
private:
    class Impl_;

public:
    Simulator(
        std::string const &device, boost::python::object dtype, std::uint_fast8_t room,
        std::uint_fast8_t baseline_grade, boost::python::object baseline_model,
        std::uint_fast8_t proposed_grade, boost::python::object proposed_model,
        unsigned long simulation_mode, std::size_t num_simulation_sets,
        std::size_t batch_size, std::size_t concurrency, boost::python::object progress);

    Simulator(Simulator const &) = delete;

    Simulator &operator=(Simulator const &) = delete;

    boost::python::list run();

private:
    std::shared_ptr<Impl_> p_impl_;
}; // class Simulator

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_SIMULATOR_HPP_INCLUDE_GUARD)
