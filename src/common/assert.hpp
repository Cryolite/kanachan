#if !defined(KANACHAN_COMMON_ASSERT_HPP_INCLUDE_GUARD)
#define KANACHAN_COMMON_ASSERT_HPP_INCLUDE_GUARD

#if defined(KANACHAN_ENABLE_ASSERT)

#include <boost/stacktrace/stacktrace.hpp>
#include <boost/current_function.hpp>
#include <boost/config.hpp>
#include <sstream>
#include <iosfwd>
#include <string_view>
#include <utility>
#include <stdexcept>


namespace Kanachan{

class AssertionFailure
  : public std::logic_error
{
public:
  explicit AssertionFailure(std::string const &error_message);

  explicit AssertionFailure(std::string_view error_message);

  AssertionFailure(AssertionFailure const &rhs) noexcept = default;

  AssertionFailure &operator=(AssertionFailure const &) noexcept = default;
}; // class AssertionFailure

namespace Detail_{

class AssertMessenger
{
public:
  explicit AssertMessenger(char const *file_name, int line_number, char const *function_name,
                           char const *expression, boost::stacktrace::stacktrace &&stacktrace);

  AssertMessenger(AssertMessenger const &) = delete;

  AssertMessenger &operator=(AssertMessenger const &) = delete;

  template<typename T>
  AssertMessenger &operator<<(T &&x)
  {
    oss_ << std::forward<T>(x);
    return *this;
  }

  AssertMessenger &operator<<(std::ostream &(*pf)(std::ostream &));

  AssertMessenger &operator<<(std::ios &(*pf)(std::ios &));

  AssertMessenger &operator<<(std::ios_base &(*pf)(std::ios_base &));

  operator int() const noexcept;

  [[noreturn]] ~AssertMessenger() noexcept(false);

private:
  std::ostringstream oss_;
  char const *file_name_;
  int line_number_;
  char const *function_name_;
  boost::stacktrace::stacktrace stacktrace_;
}; // class AssertMessenger

} // namespace Detail_

} // namespace Kanachan

#define KANACHAN_ASSERT(EXPR)                                             \
  BOOST_LIKELY(!!(EXPR)) ? 0 :                                            \
  ::Kanachan::Detail_::AssertMessenger(__FILE__,                          \
                                       __LINE__,                          \
                                       BOOST_CURRENT_FUNCTION,            \
                                       #EXPR,                             \
                                       ::boost::stacktrace::stacktrace()) \
  /**/

namespace Kanachan::Detail_{

template<typename F>
class AssertCompoundStatementExecutor
{
public:
  explicit AssertCompoundStatementExecutor(F &&f) noexcept
    : f_(std::move(f))
  {}

  AssertCompoundStatementExecutor(
    AssertCompoundStatementExecutor const &) = delete;

  AssertCompoundStatementExecutor &operator=(
    AssertCompoundStatementExecutor const &) = delete;

  ~AssertCompoundStatementExecutor() noexcept(false)
  {
    f_();
  };

private:
  F f_;
}; // class AssertCompoundStatementExecutor

class AssertCompoundStatement
{
public:
  constexpr AssertCompoundStatement() = default;

  AssertCompoundStatement(AssertCompoundStatement const &) = delete;

  AssertCompoundStatement &operator=(AssertCompoundStatement const &) = delete;

  template<typename F>
  AssertCompoundStatementExecutor<F> operator->*(F &&f) const noexcept
  {
    return AssertCompoundStatementExecutor<F>(std::move(f));
  }
}; // class AssertCompoundStatement

} // namespace Kanachan::Detail_

#define KANACHAN_ASSERT_COMPOUND_STATEMENT       \
  ::Kanachan::Detail_::AssertCompoundStatement() \
  ->* [&] () -> void                             \
  /**/

#else // defined(KANACHAN_ENABLE_ASSERT)

#include <iosfwd>

namespace Kanachan::Detail_{

class DummyAssertMessenger{
public:
  constexpr DummyAssertMessenger() = default;

  DummyAssertMessenger(DummyAssertMessenger const &) = delete;

  DummyAssertMessenger &operator=(DummyAssertMessenger const &) = delete;

  template<typename T>
  DummyAssertMessenger const &operator<<(T &&) const noexcept
  {
    return *this;
  }

  DummyAssertMessenger const &
  operator<<(std::ostream &(*)(std::ostream &)) const noexcept
  {
    return *this;
  }

  DummyAssertMessenger const &
  operator<<(std::ios &(*)(std::ios &)) const noexcept
  {
    return *this;
  }

    DummyAssertMessenger const &
    operator<<(std::ios_base &(*)(std::ios_base &)) const noexcept
  {
    return *this;
  }

  constexpr operator int() const noexcept
  {
    return 0;
  }
}; // class DummyAssertMessenger

} // namespace Kanachan::Detail_

#define KANACHAN_ASSERT(EXPR)                            \
  true ? 0 : ::Kanachan::Detail_::DummyAssertMessenger{} \
  /**/

namespace Kanachan::Detail_{

class DummyAssertCompoundStatement
{
public:
  constexpr DummyAssertCompoundStatement() = default;

  DummyAssertCompoundStatement(DummyAssertCompoundStatement const &) = delete;

  DummyAssertCompoundStatement &operator=(
    DummyAssertCompoundStatement const &) = delete;

  template<typename F>
  void operator->*(F &&) const noexcept
  {}
}; // class DummyAssertCompoundStatement

} // namespace Kanachan::Detail_

#define KANACHAN_ASSERT_COMPOUND_STATEMENT            \
  ::Kanachan::Detail_::DummyAssertCompoundStatement() \
  ->* [&] () -> void                                  \
  /**/

#endif // defined(KANACHAN_ENABLE_ASSERT)

#endif // !defined(KANACHAN_COMMON_ASSERT_HPP_INCLUDE_GUARD)
