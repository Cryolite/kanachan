#if !defined(KANACHAN_SIMULATION_UTILITY_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_UTILITY_HPP_INCLUDE_GUARD

#include <boost/python/object.hpp>
#include <vector>
#include <cstdint>


namespace Kanachan{

std::vector<std::uint_least32_t> getRandomSeed();

bool hasattr(boost::python::object o, char const *name);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_UTILITY_HPP_INCLUDE_GUARD)
