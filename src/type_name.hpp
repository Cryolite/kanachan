#if !defined(KANACHAN_TYPE_NAME_HPP_INCLUDE_GUARD)
#define KANACHAN_TYPE_NAME_HPP_INCLUDE_GUARD

#include <string>
#include <typeinfo>


namespace Kanachan{

std::string getTypeName(std::type_info const &ti);

template<typename T>
std::string getTypeName(T const &x)
{
  return Kanachan::getTypeName(typeid(x));
}

} // namespace Kanachan

#endif // !defined(KANACHAN_TYPE_NAME_HPP_INCLUDE_GUARD)
