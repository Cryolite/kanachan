#if !defined(KANACHAN_SIMULATION_PAISHAN_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_PAISHAN_HPP_INCLUDE_GUARD

#include <boost/python/list.hpp>
#include <vector>
#include <array>
#include <cstdint>


namespace Kanachan{

class Paishan;

void swap(Paishan &lhs, Paishan &rhs) noexcept;

class Paishan
{
public:
  explicit Paishan(std::vector<std::uint_least32_t> const &seed);

  explicit Paishan(boost::python::list paishan);

  Paishan(Paishan const &rhs) = default;

  Paishan(Paishan &&rhs) = default;

  void swap(Paishan &rhs) noexcept;

  Paishan &operator=(Paishan const &rhs);

  Paishan &operator=(Paishan &&rhs) noexcept;

public:
  std::uint_fast8_t operator[](std::uint_fast8_t index) const;

private:
  std::array<std::uint_fast8_t, 136u> tiles_;
}; // class Paishan

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_PAISHAN_HPP_INCLUDE_GUARD)
