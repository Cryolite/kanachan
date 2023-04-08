#if !defined(KANACHAN_COMMON_THREAD_HPP_INCLUDE_GUARD)
#define KANACHAN_COMMON_THREAD_HPP_INCLUDE_GUARD

#include <string>
#include <stdexcept>


namespace Kanachan{

class ThreadTermination
    : public std::runtime_error
{
public:
    explicit ThreadTermination(char const *message);

    explicit ThreadTermination(std::string const &message);
}; // class ThreadTermination

} // namespace Kanachan

#endif // !defined(KANACHAN_COMMON_THREAD_HPP_INCLUDE_GUARD)
