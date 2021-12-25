#if !defined(KANACHAN_SIMULATION_XIANGTING_CALCULATOR_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_XIANGTING_CALCULATOR_HPP_INCLUDE_GUARD

#include <marisa.h>
#include <vector>
#include <utility>
#include <string>
#include <cstdint>


namespace Kanachan{

class XiangtingCalculator
{
public:
  XiangtingCalculator();

private:
  std::uint_fast8_t shupaiImpl_(
    std::string const &key, bool const headless) const;

  std::uint_fast8_t zipaiImpl_(
    std::string const &key, bool const headless) const;

  template<typename RandomAccessIterator>
  static std::uint_fast8_t qiduiziImpl_(RandomAccessIterator first);

  template<typename RandomAccessIterator>
  static std::uint_fast8_t shisanyaoImpl_(RandomAccessIterator first);

public:
  template<typename RandomAccessIterator>
  std::uint_fast8_t operator()(
    RandomAccessIterator first, RandomAccessIterator last,
    std::uint_fast8_t n) const;

  template<typename RandomAccessRange>
  std::uint_fast8_t operator()(
    RandomAccessRange const &tiles, std::uint_fast8_t n) const;

private:
  marisa::Trie shupai_trie_;
  std::vector<std::pair<std::uint_fast8_t, std::uint_fast8_t> > shupai_xiangting_;
  marisa::Trie zipai_trie_;
  std::vector<std::pair<std::uint_fast8_t, std::uint_fast8_t> > zipai_xiangting_;
}; // class XiangtingCalculator

template<typename RandomAccessRange>
std::uint_fast8_t calculateXiangting(
  RandomAccessRange const &tiles, std::uint_fast8_t n);

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_XIANGTING_CALCULATOR_HPP_INCLUDE_GUARD)
