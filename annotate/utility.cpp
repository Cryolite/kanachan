#include "utility.hpp"

#include "throw.hpp"
#include <string_view>
#include <string>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>


namespace Kanachan{

namespace{

using std::placeholders::_1;

} // namepsace *unnamed*

std::uint_fast8_t pai2Num(std::string_view const &pai)
{
  if (pai.size() != 2u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "pai = " << pai;
  }

  std::uint_fast8_t base = std::numeric_limits<std::uint_fast8_t>::max();
  switch (pai[1u]) {
  case 'm':
    base = 0u;
    break;
  case 'p':
    base = 10u;
    break;
  case 's':
    base = 20u;
    break;
  case 'z':
    base = 30u;
    break;
  default:
    KANACHAN_THROW<std::invalid_argument>(_1) << "pai = " << pai;
  }

  std::uint_fast8_t num = pai[0u] - '0';
  if (pai[1u] == 'z') {
    if (pai[0u] == '0' || pai[0u] >= '8') {
      KANACHAN_THROW<std::invalid_argument>(_1) << "pai = " << pai;
    }
    --num;
  }

  return base + num;
}

std::uint_fast8_t pai2Num(std::string const &pai)
{
  return pai2Num(std::string_view(pai.cbegin(), pai.cend()));
}

} // namespace Kanachan
