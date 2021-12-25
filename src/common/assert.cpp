#include "assert.hpp"

#if defined(KANACHAN_ENABLE_ASSERT)

#include "throw.hpp"
#include <boost/exception/enable_error_info.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/info.hpp>
#include <boost/stacktrace/stacktrace.hpp>
#include <ostream>
#include <ios>
#include <string_view>
#include <utility>
#include <stdexcept>


#if defined(KANACHAN_WITH_COVERAGE)

extern "C" void __gcov_flush();

#endif // defined(KANACHAN_WITH_COVERAGE)

namespace Kanachan{

AssertionFailure::AssertionFailure(std::string const &error_message)
  : std::logic_error(error_message)
{}

AssertionFailure::AssertionFailure(std::string_view error_message)
  : AssertionFailure(std::string(error_message))
{}

namespace Detail_{

AssertMessenger::AssertMessenger(char const *file_name, int line_number, char const *function_name,
                                 char const *expression, boost::stacktrace::stacktrace &&stacktrace)
  : oss_(),
    file_name_(file_name),
    line_number_(line_number),
    function_name_(function_name),
    stacktrace_(std::move(stacktrace))
{
  oss_ << file_name_ << ':' << line_number_ << ": " << function_name_ << ": "
       << "Assertion `" << expression << "' failed.\n";
}

AssertMessenger &AssertMessenger::operator<<(std::ostream &(*pf)(std::ostream &))
{
  oss_ << pf;
  return *this;
}

AssertMessenger &AssertMessenger::operator<<(std::ios &(*pf)(std::ios &))
{
  oss_ << pf;
  return *this;
}

AssertMessenger &AssertMessenger::operator<<(std::ios_base &(*pf)(std::ios_base &))
{
  oss_ << pf;
  return *this;
}

AssertMessenger::operator int() const noexcept
{
  return 0;
}

[[noreturn]] AssertMessenger::~AssertMessenger() noexcept(false)
{
  throw boost::enable_error_info(Kanachan::AssertionFailure(oss_.str()))
    << boost::throw_file(file_name_)
    << boost::throw_line(line_number_)
    << boost::throw_function(function_name_)
    << Kanachan::StackTraceErrorInfo(std::move(stacktrace_));
}

} // namespace Detail_

} // namespace Kanachan

#endif // defined(KANACHAN_ENABLE_ASSERT)
