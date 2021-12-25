#if !defined(KANACHAN_COMMON_THROW_HPP_INCLUDE_GUARD)
#define KANACHAN_COMMON_THROW_HPP_INCLUDE_GUARD

#include <boost/exception/enable_error_info.hpp>
#include <boost/exception/info.hpp>
#include <boost/exception/exception.hpp>
#include <boost/stacktrace/stacktrace.hpp>
#include <boost/current_function.hpp>
#include <sstream>
#include <ostream>
#include <ios>
#include <string>
#include <type_traits>
#include <functional>
#include <tuple>
#include <utility>
#include <exception>


namespace Kanachan{

namespace Detail_{

struct StackTraceErrorInfoTag_;

} // namespace Detail_

using StackTraceErrorInfo = boost::error_info<
  Detail_::StackTraceErrorInfoTag_, boost::stacktrace::stacktrace>;

namespace Detail_{

enum struct ThrowType
{
  throw_,
  throw_with_nested,
}; // enum struct ThrowType

template<typename Exception, Kanachan::Detail_::ThrowType throw_type,
         typename HasPlaceholder, typename... Args>
class ExceptionThrower;

template<typename Exception, Kanachan::Detail_::ThrowType throw_type, typename... Args>
class ExceptionThrower<Exception, throw_type, std::false_type, Args...>
{
private:
  static_assert(!std::is_reference_v<Exception>);
  static_assert(!std::is_const_v<Exception>);
  static_assert(!std::is_volatile_v<Exception>);
  static_assert((... && std::is_reference_v<Args>));

private:
  static Exception constructException(Args... args)
  noexcept(noexcept(Exception(std::forward<Args>(args)...)))
  {
    return Exception(std::forward<Args>(args)...);
  }

public:
  ExceptionThrower(char const *function_name, char const *file_name, int line_number,
                   boost::stacktrace::stacktrace &&stacktrace, std::tuple<Args...> args) noexcept
    : function_name_(function_name),
      file_name_(file_name),
      line_number_(line_number),
      stacktrace_(std::move(stacktrace)),
      args_(std::move(args))
  {}

  ExceptionThrower(ExceptionThrower const &) = delete;

  ExceptionThrower &operator=(ExceptionThrower const &) = delete;

public:
  [[noreturn]] ~ExceptionThrower() noexcept(false)
  {
    if constexpr (throw_type == Kanachan::Detail_::ThrowType::throw_) {
      throw boost::enable_error_info(std::apply(&ExceptionThrower::constructException, std::move(args_)))
        << boost::throw_function(function_name_)
        << boost::throw_file(file_name_)
        << boost::throw_line(line_number_)
        << StackTraceErrorInfo(std::move(stacktrace_));
    }
    else {
      std::throw_with_nested(
        boost::enable_error_info(std::apply(&ExceptionThrower::constructException, std::move(args_)))
          << boost::throw_function(function_name_)
          << boost::throw_file(file_name_)
          << boost::throw_line(line_number_)
          << StackTraceErrorInfo(std::move(stacktrace_)));
    }
  }

private:
  char const *function_name_;
  char const *file_name_;
  int line_number_;
  boost::stacktrace::stacktrace stacktrace_;
  std::tuple<Args...> args_;
}; // class ExceptionThrower

template<typename Exception, Kanachan::Detail_::ThrowType throw_type, typename... Args>
class ExceptionThrower<Exception, throw_type, std::true_type, Args...>
{
private:
  static_assert(!std::is_reference_v<Exception>);
  static_assert(!std::is_const_v<Exception>);
  static_assert(!std::is_volatile_v<Exception>);
  static_assert((... && std::is_reference_v<Args>));

private:
  template<typename T>
  using RemoveCVRef = std::remove_cv_t<std::remove_reference_t<T>>;

  template<typename T>
  static constexpr bool is_placeholder_v = (std::is_placeholder<RemoveCVRef<T>>::value == 1);

  template<typename T, typename U>
  static std::enable_if_t<!is_placeholder_v<T>, T &&>
  substitutePlaceholder(T &&arg, U &&subst) noexcept
  {
    return arg;
  }

  template<typename T, typename U>
  static std::enable_if_t<is_placeholder_v<T>, U &&>
  substitutePlaceholder(T &&arg, U &&substitute) noexcept
  {
    return substitute;
  }

  static Exception constructException(std::string const &what, Args... args)
  noexcept(noexcept(Exception(substitutePlaceholder(std::forward<Args>(args), what)...)))
  {
    return Exception(substitutePlaceholder(std::forward<Args>(args), what)...);
  }

public:
  ExceptionThrower(char const *function_name, char const *file_name, int line_number,
                   boost::stacktrace::stacktrace &&stacktrace, std::tuple<Args...> args) noexcept
    : function_name_(function_name),
      file_name_(file_name),
      line_number_(line_number),
      stacktrace_(std::move(stacktrace)),
      args_(std::move(args)),
      oss_()
  {}

  ExceptionThrower(ExceptionThrower const &) = delete;

  ExceptionThrower &operator=(ExceptionThrower const &) = delete;

  template<typename T>
  ExceptionThrower &operator<<(T &&value)
  {
    oss_ << std::forward<T>(value);
    return *this;
  }

  ExceptionThrower &operator<<(std::ostream &(*pf)(std::ostream &))
  {
    oss_ << pf;
    return *this;
  }

  ExceptionThrower &operator<<(std::ios &(*pf)(std::ios &))
  {
    oss_ << pf;
    return *this;
  }

  ExceptionThrower &operator<<(std::ios_base &(*pf)(std::ios_base &))
  {
    oss_ << pf;
    return *this;
  }

  [[noreturn]] ~ExceptionThrower() noexcept(false)
  {
    if constexpr (throw_type == Kanachan::Detail_::ThrowType::throw_) {
      throw boost::enable_error_info(
        std::apply(&ExceptionThrower::constructException,
                   std::tuple_cat(std::forward_as_tuple(oss_.str()), std::move(args_))))
        << boost::throw_function(function_name_)
        << boost::throw_file(file_name_)
        << boost::throw_line(line_number_)
        << StackTraceErrorInfo(std::move(stacktrace_));
    }
    else {
      std::throw_with_nested(
        boost::enable_error_info(
          std::apply(&ExceptionThrower::constructException,
                     std::tuple_cat(std::forward_as_tuple(oss_.str()), std::move(args_))))
          << boost::throw_function(function_name_)
          << boost::throw_file(file_name_)
          << boost::throw_line(line_number_)
          << StackTraceErrorInfo(std::move(stacktrace_)));
    }
  }

private:
  char const *function_name_;
  char const *file_name_;
  int line_number_;
  boost::stacktrace::stacktrace stacktrace_;
  std::tuple<Args...> args_;
  std::ostringstream oss_;
}; // class ExceptionThrower

class ExceptionInfoHolder
{
public:
  ExceptionInfoHolder(char const *function_name, char const *file_name, int line_number,
                      boost::stacktrace::stacktrace &&stacktrace) noexcept
    : function_name_(function_name),
      file_name_(file_name),
      line_number_(line_number),
      stacktrace_(std::move(stacktrace))
  {}

  ExceptionInfoHolder(ExceptionInfoHolder const &) = delete;

  ExceptionInfoHolder &operator=(ExceptionInfoHolder const &) = delete;

  template<typename T>
  using RemoveCVRef = std::remove_cv_t<std::remove_reference_t<T>>;

  template<typename... Args>
  using HasPlaceholder = std::bool_constant<(... || (std::is_placeholder<RemoveCVRef<Args>>::value == 1))>;

  template<typename Exception, Kanachan::Detail_::ThrowType throw_type, typename... Args>
  using Thrower = ExceptionThrower<Exception, throw_type, HasPlaceholder<Args...>, Args &&...>;

  template<typename Exception, typename... Args>
  Thrower<Exception, Kanachan::Detail_::ThrowType::throw_, Args...>
  setExceptionToThrow(Args &&... args) noexcept
  {
    static_assert(!std::is_reference_v<Exception>);
    static_assert(!std::is_const_v<Exception>);
    static_assert(!std::is_volatile_v<Exception>);
    static_assert((... && (std::is_placeholder<RemoveCVRef<Args>>::value <= 1)));
    using Result = Thrower<Exception, Kanachan::Detail_::ThrowType::throw_, Args...>;
    return Result(function_name_, file_name_, line_number_, std::move(stacktrace_),
                  std::forward_as_tuple(std::forward<Args>(args)...));
  }

  template<typename Exception, typename... Args>
  Thrower<Exception, Kanachan::Detail_::ThrowType::throw_with_nested, Args...>
  setExceptionToThrowWithNested(Args &&... args) noexcept
  {
    static_assert(!std::is_reference_v<Exception>);
    static_assert(!std::is_const_v<Exception>);
    static_assert(!std::is_volatile_v<Exception>);
    static_assert((... && (std::is_placeholder<RemoveCVRef<Args>>::value <= 1)));
    using Result = Thrower<Exception, Kanachan::Detail_::ThrowType::throw_with_nested, Args...>;
    return Result(function_name_, file_name_, line_number_, std::move(stacktrace_),
                  std::forward_as_tuple(std::forward<Args>(args)...));
  }

private:
  char const *function_name_;
  char const *file_name_;
  int line_number_;
  boost::stacktrace::stacktrace stacktrace_;
}; // class ExceptionInfoHolder

} // namespace Detail_

} // namespace Kanachan

#define KANACHAN_THROW                      \
  ::Kanachan::Detail_::ExceptionInfoHolder( \
    BOOST_CURRENT_FUNCTION,                 \
    __FILE__,                               \
    __LINE__,                               \
    ::boost::stacktrace::stacktrace())      \
    .template setExceptionToThrow           \
  /**/

#define KANACHAN_THROW_WITH_NESTED          \
  ::Kanachan::Detail_::ExceptionInfoHolder( \
    BOOST_CURRENT_FUNCTION,                 \
    __FILE__,                               \
    __LINE__,                               \
    ::boost::stacktrace::stacktrace())      \
    .template setExceptionToThrowWithNested \
  /**/

namespace Kanachan::Detail_{

class TerminateHandlerSetter
{
private:
  [[noreturn]] static void terminate_handler_() noexcept;

public:
  TerminateHandlerSetter() noexcept;

  TerminateHandlerSetter(TerminateHandlerSetter const &) = delete;

  TerminateHandlerSetter &operator=(TerminateHandlerSetter const &) = delete;
}; // class TerminateHandlerSetter

// The initialization of this variable triggers the establishment of a function
// provided by this program as a customized handler function for terminating
// exception processing. If the following variable definition (not declaration)
// moves to a .cpp file, the initialization may not execute unless some
// function that compiles in the same translation unit is called.
inline TerminateHandlerSetter terminate_handler_setter;

} // namespace Kanachan::Detail_

#endif // !defined(KANACHAN_COMMON_THROW_HPP_INCLUDE_GUARD)
