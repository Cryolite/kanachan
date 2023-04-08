#if !defined(KANACHAN_SIMULATION_UTILITY_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_UTILITY_HPP_INCLUDE_GUARD

#include <vector>
#include <cstdint>


namespace Kanachan{

std::vector<std::uint_least32_t> getRandomSeed();

[[noreturn]] void translatePythonException();

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_UTILITY_HPP_INCLUDE_GUARD)
