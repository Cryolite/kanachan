#if !defined(KANACHAN_ANNOTATION_UTILITY_HPP_INCLUDE_GUARD)
#define KANACHAN_ANNOTATION_UTILITY_HPP_INCLUDE_GUARD

#include "common/throw.hpp"
#include <iostream>
#include <string_view>
#include <string>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

std::uint_fast8_t pai2Num(std::string_view const &pai);

std::uint_fast8_t pai2Num(std::string const &pai);

namespace{

constexpr std::uint_fast64_t encodeChi_(
  std::uint_fast8_t const a, std::uint_fast8_t const b, std::uint_fast8_t const c)
{
  return (1ul << a) + (1ul << b) + (1ul << (c + 10u));
}

} // namespace *unnamed*

template<typename Iterator>
std::uint_fast8_t encodeChi(Iterator first, Iterator last)
{
  std::array<std::uint_fast8_t, 3u> numbers{
    *first++,
    *first++,
    *first++
  };
  if (first != last) {
    KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
  }

  std::uint_fast8_t const color = numbers[0u] / 10u;
  for (auto &e : numbers) {
    e %= 10u;
  }

  std::uint_fast8_t encode = std::numeric_limits<std::uint_fast8_t>::max();
  switch ((1ul << numbers[0u]) + (1ul << numbers[1u]) + (1ul << (numbers[2u] + 10u))) {
  case encodeChi_(2u, 3u, 1u):
    encode = 0u;
    break;
  case encodeChi_(1u, 3u, 2u):
    encode = 1u;
    break;
  case encodeChi_(3u, 4u, 2u):
    encode = 2u;
    break;
  case encodeChi_(1u, 2u, 3u):
    encode = 3u;
    break;
  case encodeChi_(2u, 4u, 3u):
    encode = 4u;
    break;
  case encodeChi_(4u, 5u, 3u):
    encode = 5u;
    break;
  case encodeChi_(4u, 0u, 3u):
    encode = 6u;
    break;
  case encodeChi_(2u, 3u, 4u):
    encode = 7u;
    break;
  case encodeChi_(3u, 5u, 4u):
    encode = 8u;
    break;
  case encodeChi_(3u, 0u, 4u):
    encode = 9u;
    break;
  case encodeChi_(5u, 6u, 4u):
    encode = 10u;
    break;
  case encodeChi_(0u, 6u, 4u):
    encode = 11u;
    break;
  case encodeChi_(3u, 4u, 5u):
    encode = 12u;
    break;
  case encodeChi_(3u, 4u, 0u):
    encode = 13u;
    break;
  case encodeChi_(4u, 6u, 5u):
    encode = 14u;
    break;
  case encodeChi_(4u, 6u, 0u):
    encode = 15u;
    break;
  case encodeChi_(6u, 7u, 5u):
    encode = 16u;
    break;
  case encodeChi_(6u, 7u, 0u):
    encode = 17u;
    break;
  case encodeChi_(4u, 5u, 6u):
    encode = 18u;
    break;
  case encodeChi_(4u, 0u, 6u):
    encode = 19u;
    break;
  case encodeChi_(5u, 7u, 6u):
    encode = 20u;
    break;
  case encodeChi_(0u, 7u, 6u):
    encode = 21u;
    break;
  case encodeChi_(7u, 8u, 6u):
    encode = 22u;
    break;
  case encodeChi_(5u, 6u, 7u):
    encode = 23u;
    break;
  case encodeChi_(0u, 6u, 7u):
    encode = 24u;
    break;
  case encodeChi_(6u, 8u, 7u):
    encode = 25u;
    break;
  case encodeChi_(8u, 9u, 7u):
    encode = 26u;
    break;
  case encodeChi_(6u, 7u, 8u):
    encode = 27u;
    break;
  case encodeChi_(7u, 9u, 8u):
    encode = 28u;
    break;
  case encodeChi_(7u, 8u, 9u):
    encode = 29u;
    break;
  default:
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }

  return 30u * color + encode;
}

inline std::uint_fast8_t encodeChi(
  std::uint_fast8_t a, std::uint_fast8_t b, std::uint_fast8_t c)
{
  std::array<std::uint_fast8_t, 3u> tiles{a, b, c};
  return encodeChi(tiles.cbegin(), tiles.cend());
}

namespace{

constexpr std::uint_fast64_t encodePeng_(
  std::uint_fast8_t const a, std::uint_fast8_t const b, std::uint_fast8_t const c)
{
  return (1ul << a) + (1ul << (b + 10u)) + (1ul << (c + 20u));
}

} // namespace *unnamed*

template<typename Iterator>
std::uint_fast8_t encodePeng(Iterator first, Iterator last)
{
  std::array<std::uint_fast8_t, 3u> numbers{
    *first++,
    *first++,
    *first++
  };
  if (first != last) {
    KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
  }

  std::uint_fast8_t encode = std::numeric_limits<std::uint_fast8_t>::max();
  switch ((1ul << numbers[0u]) + (1ul << (numbers[1u] + 10u)) + (1ul << (numbers[2u] + 20u))) {
  case encodePeng_(1u, 1u, 1u):
    encode = 0u;
    break;
  case encodePeng_(2u, 2u, 2u):
    encode = 1u;
    break;
  case encodePeng_(3u, 3u, 3u):
    encode = 2u;
    break;
  case encodePeng_(4u, 4u, 4u):
    encode = 3u;
    break;
  case encodePeng_(5u, 5u, 5u):
    encode = 4u;
    break;
  case encodePeng_(0u, 5u, 5u):
    encode = 5u;
    break;
  case encodePeng_(5u, 5u, 0u):
    encode = 6u;
    break;
  case encodePeng_(6u, 6u, 6u):
    encode = 7u;
    break;
  case encodePeng_(7u, 7u, 7u):
    encode = 8u;
    break;
  case encodePeng_(8u, 8u, 8u):
    encode = 9u;
    break;
  case encodePeng_(9u, 9u, 9u):
    encode = 10u;
    break;
  case encodePeng_(11u, 11u, 11u):
    encode = 11u;
    break;
  case encodePeng_(12u, 12u, 12u):
    encode = 12u;
    break;
  case encodePeng_(13u, 13u, 13u):
    encode = 13u;
    break;
  case encodePeng_(14u, 14u, 14u):
    encode = 14u;
    break;
  case encodePeng_(15u, 15u, 15u):
    encode = 15u;
    break;
  case encodePeng_(10u, 15u, 15u):
    encode = 16u;
    break;
  case encodePeng_(15u, 15u, 10u):
    encode = 17u;
    break;
  case encodePeng_(16u, 16u, 16u):
    encode = 18u;
    break;
  case encodePeng_(17u, 17u, 17u):
    encode = 19u;
    break;
  case encodePeng_(18u, 18u, 18u):
    encode = 20u;
    break;
  case encodePeng_(19u, 19u, 19u):
    encode = 21u;
    break;
  case encodePeng_(21u, 21u, 21u):
    encode = 22u;
    break;
  case encodePeng_(22u, 22u, 22u):
    encode = 23u;
    break;
  case encodePeng_(23u, 23u, 23u):
    encode = 24u;
    break;
  case encodePeng_(24u, 24u, 24u):
    encode = 25u;
    break;
  case encodePeng_(25u, 25u, 25u):
    encode = 26u;
    break;
  case encodePeng_(20u, 25u, 25u):
    encode = 27u;
    break;
  case encodePeng_(25u, 25u, 20u):
    encode = 28u;
    break;
  case encodePeng_(26u, 26u, 26u):
    encode = 29u;
    break;
  case encodePeng_(27u, 27u, 27u):
    encode = 30u;
    break;
  case encodePeng_(28u, 28u, 28u):
    encode = 31u;
    break;
  case encodePeng_(29u, 29u, 29u):
    encode = 32u;
    break;
  case encodePeng_(30u, 30u, 30u):
    encode = 33u;
    break;
  case encodePeng_(31u, 31u, 31u):
    encode = 34u;
    break;
  case encodePeng_(32u, 32u, 32u):
    encode = 35u;
    break;
  case encodePeng_(33u, 33u, 33u):
    encode = 36u;
    break;
  case encodePeng_(34u, 34u, 34u):
    encode = 37u;
    break;
  case encodePeng_(35u, 35u, 35u):
    encode = 38u;
    break;
  case encodePeng_(36u, 36u, 36u):
    encode = 39u;
    break;
  default:
    KANACHAN_THROW<std::runtime_error>(std::placeholders::_1)
      << '(' << static_cast<unsigned>(numbers[0u]) << ", "
      << static_cast<unsigned>(numbers[1u]) << ", "
      << static_cast<unsigned>(numbers[2u]) << "): A broken data.";
  }

  return encode;
}

inline std::uint_fast8_t encodePeng(
  std::uint_fast8_t a, std::uint_fast8_t b, std::uint_fast8_t c)
{
  std::array<std::uint_fast8_t, 3u> tiles{a, b, c};
  return encodePeng(tiles.cbegin(), tiles.cend());
}

template<typename T>
class CategoryEncoder
{
public:
  CategoryEncoder(T const &category, T const &num_categories)
    : category_(category),
      num_categories_(num_categories)
  {
    if (category < 0) {
      KANACHAN_THROW<std::invalid_argument>(std::placeholders::_1)
        << "category = " << static_cast<int>(category);
    }
    if (category != std::numeric_limits<T>::max() && category >= num_categories) {
      KANACHAN_THROW<std::invalid_argument>(std::placeholders::_1)
        << "category = " << static_cast<long>(category)
        << ", # of categories = " << static_cast<long>(num_categories);
    }
  }

  friend std::ostream &operator<<(std::ostream &os, CategoryEncoder const &rhs)
  {
    if (rhs.category_ == std::numeric_limits<T>::max()) {
      os << '0';
      for (std::size_t i = 1u; i < rhs.num_categories_; ++i) {
        os << ",0";
      }
      return os;
    }

    if (rhs.category_ == 0) {
      os << '1';
      for (std::size_t i = 1u; i < rhs.num_categories_; ++i) {
        os << ",0";
      }
    }
    else {
      os << '0';
      for (std::size_t i = 1u; i < rhs.category_; ++i) {
        os << ",0";
      }
      os << ",1";
      for (std::size_t i = rhs.category_ + 1; i < rhs.num_categories_; ++i) {
        os << ",0";
      }
    }
    return os;
  }

private:
  T category_;
  T num_categories_;
}; // class CategoryEncoder

template<typename T, typename U>
CategoryEncoder<T> encodeCategory(T const &category, U const &num_categories)
{
  if (std::numeric_limits<T>::max() < std::numeric_limits<U>::max()) {
    if (num_categories >= std::numeric_limits<T>::max() - 1) {
      KANACHAN_THROW<std::invalid_argument>("");
    }
  }
  if (category < 0) {
    KANACHAN_THROW<std::invalid_argument>(std::placeholders::_1)
      << "category = " << static_cast<int>(category);
  }
  if (category != std::numeric_limits<T>::max() && category >= num_categories) {
    KANACHAN_THROW<std::invalid_argument>(std::placeholders::_1)
      << "category = " << static_cast<long>(category)
      << ", # of categories" << static_cast<long>(num_categories);
  }

  return CategoryEncoder<T>(category, num_categories);
}

} // namespace Kanachan

#endif // !defined(KANACHAN_ANNOTATION_UTILITY_HPP_INCLUDE_GUARD)
