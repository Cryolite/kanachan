#include <string>
#include <memory>
#include <typeinfo>
#include <cstdlib>
#include <cxxabi.h>


namespace Kanachan{

std::string getTypeName(std::type_info const &ti)
{
  char const * const name = ti.name();
  int status;
  std::unique_ptr<char, void (*)(void *)> p(
    abi::__cxa_demangle(name, NULL, NULL, &status), std::free);
  return { status == 0 ? p.get() : name };
}

} // namespace Kanachan
