#include "common/throw.hpp"
#include "common/type_name.hpp"
#include <boost/exception/get_error_info.hpp>
#include <boost/exception/exception.hpp>
#include <boost/stacktrace/stacktrace.hpp>
#include <iostream>
#include <vector>
#include <exception>
#include <cstdlib>


#if defined(KANACHAN_WITH_COVERAGE)

extern "C" void __gcov_flush();

#endif // defined(KANACHAN_WITH_COVERAGE)

namespace Kanachan::Detail_{

namespace{

void printErrorMessage(boost::exception const &e)
{
  if (char const * const * const p = boost::get_error_info<boost::throw_file>(e)) {
    std::cerr << *p << ':';
  }
  if (int const * const p = boost::get_error_info<boost::throw_line>(e)) {
    std::cerr << *p << ": ";
  }
  if (char const * const * const p = boost::get_error_info<boost::throw_function>(e)) {
    std::cerr << *p << ": ";
  }
  if (std::exception const * const p = dynamic_cast<std::exception const *>(&e)) {
    std::cerr << p->what() << '\n';
  }
}

} // namespace *unnamed*

[[noreturn]] void TerminateHandlerSetter::terminate_handler_() noexcept
try {
  using Stacktrace = boost::stacktrace::stacktrace;

  std::vector<std::exception_ptr> nested_exceptions;
  {
    std::exception_ptr const p = std::current_exception();
    if (p == nullptr) {
      std::cerr << "`std::terminate' is called without throwing any exception.\n";
      Stacktrace stacktrace;
      if (!stacktrace.empty()) {
        std::cerr << "Backtrace:\n" << stacktrace;
      }
#if defined(KANACHAN_WITH_COVERAGE)
      __gcov_flush(); std::abort();
#else // defined(KANACHAN_WITH_COVERAGE)
      std::abort();
#endif // defined(KANACHAN_WITH_COVERAGE)
    }
    nested_exceptions.push_back(p);
  }
  for (;;) {
    try {
      std::rethrow_exception(nested_exceptions.back());
    }
    catch (boost::exception const &e) {
      try {
        std::rethrow_if_nested(e);
        break;
      }
      catch (...) {
        std::exception_ptr const p = std::current_exception();
        nested_exceptions.push_back(p);
      }
    }
    catch (std::exception const &e) {
      try {
        std::rethrow_if_nested(e);
        break;
      }
      catch (...) {
        std::exception_ptr const p = std::current_exception();
        nested_exceptions.push_back(p);
      }
    }
  }

  try {
    std::rethrow_exception(nested_exceptions.back());
  }
  catch (boost::exception const &e) {
    std::cerr << "`std::terminate' is called after throwing an instance of `"
              << Kanachan::getTypeName(e) << "'.\n";
    printErrorMessage(e);
    if (Stacktrace const * const p = boost::get_error_info<Kanachan::StackTraceErrorInfo>(e)) {
      if (p->size() != 0) {
        std::cerr << "Backtrace:\n" << *p;
      }
    }
  }
  catch (std::exception const &e) {
    std::cerr << "`std::terminate' is called after throwing an instance of `"
              << Kanachan::getTypeName(e) << "'.\n" << e.what() << '\n';
  }
  catch (...) {
    std::cerr << "`std::terminate' is called after throwing an instance of an unknown type.\n";
  }
  nested_exceptions.pop_back();

  while (!nested_exceptions.empty()) {
    try {
      std::rethrow_exception(nested_exceptions.back());
    }
    catch (boost::exception const &e) {
      std::cerr << "A nesting exception of type `" << Kanachan::getTypeName(e) << "'.\n";
      printErrorMessage(e);
    }
    catch (std::exception const &e) {
      std::cerr << "A nesting exception of type `" << Kanachan::getTypeName(e) << "'.\n";
      std::cerr << e.what() << '\n';
    }
    catch (...) {
      std::cerr << "A nesting exception of an unknown type.\n";
    }
    nested_exceptions.pop_back();
  }

  std::cerr << std::flush;
#if defined(KANACHAN_WITH_COVERAGE)
  __gcov_flush(); std::abort();
#else // defined(KANACHAN_WITH_COVERAGE)
  std::abort();
#endif // defined(KANACHAN_WITH_COVERAGE)
}
catch (...) {
#if defined(KANACHAN_WITH_COVERAGE)
  __gcov_flush(); std::abort();
#else // defined(KANACHAN_WITH_COVERAGE)
  std::abort();
#endif // defined(KANACHAN_WITH_COVERAGE)
}

TerminateHandlerSetter::TerminateHandlerSetter() noexcept
{
  std::set_terminate(&terminate_handler_);
}

} // namespace Kanachan::Detail_
