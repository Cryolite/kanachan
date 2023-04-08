#include "common/thread.hpp"

#include <string>
#include <stdexcept>


namespace Kanachan{

Kanachan::ThreadTermination::ThreadTermination(char const *message)
    : std::runtime_error(message)
{}

Kanachan::ThreadTermination::ThreadTermination(std::string const &message)
    : std::runtime_error(message)
{}

} // namespace Kanachan
