#if !defined(KANACHAN_UTILITY_HPP_INCLUDE_GUARD)
#define KANACHAN_UTILITY_HPP_INCLUDE_GUARD

#include "throw.hpp"
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

#endif // !defined(KANACHAN_UTILITY_HPP_INCLUDE_GUARD)
