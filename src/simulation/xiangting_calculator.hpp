#if !defined(KANACHAN_SIMULATION_XIANGTING_CALCULATOR_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_XIANGTING_CALCULATOR_HPP_INCLUDE_GUARD

#include <boost/python/list.hpp>
#include <boost/python/long.hpp>
#include <filesystem>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>


namespace Kanachan{

class XiangtingCalculator
{
public:
  explicit XiangtingCalculator(
    std::filesystem::path const &prefix = std::filesystem::path("/home/ubuntu/.local/share/kanachan"));

  explicit XiangtingCalculator(std::string const &prefix);

  template<typename RandomAccessIterator>
  std::uint_fast8_t operator()(
    RandomAccessIterator first, RandomAccessIterator last,
    std::uint_fast8_t n) const;

  template<typename RandomAccessRange>
  std::uint_fast8_t operator()(
    RandomAccessRange const &tiles, std::uint_fast8_t n) const;

  boost::python::long_ calculate(
    boost::python::list tile_counts, boost::python::long_ n) const;

private:
  class Impl_;
  std::shared_ptr<Impl_> p_impl_;
}; // class XiangtingCalculator

template<typename RandomAccessRange>
std::uint_fast8_t calculateXiangting(
  RandomAccessRange const &tiles, std::uint_fast8_t n);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_XIANGTING_CALCULATOR_HPP_INCLUDE_GUARD)
