#if !defined(KANACHAN_SIMULATION_GIL_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_GIL_HPP_INCLUDE_GUARD

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <mutex>
#include <thread>
#include <stop_token>
#include <chrono>


namespace Kanachan::GIL{

class RecursiveLock
{
public:
    RecursiveLock();

    RecursiveLock(RecursiveLock const &) = delete;

    RecursiveLock(RecursiveLock &&rhs) noexcept;

    ~RecursiveLock();

    RecursiveLock &operator=(RecursiveLock const &) = delete;

    RecursiveLock &operator=(RecursiveLock &&rhs) noexcept;

private:
    PyGILState_STATE state_;
    bool owns_;
}; // class RecursiveLock

class RecursiveRelease
{
public:
    RecursiveRelease();

    RecursiveRelease(RecursiveRelease const &) = delete;

    ~RecursiveRelease();

    RecursiveRelease &operator=(RecursiveRelease const &) = delete;

private:
    PyThreadState *save_;
}; // class UniqueRelease

} // namespace Kanachan::GIL

#endif // !defined(KANACHAN_SIMULATION_GIL_HPP_INCLUDE_GUARD)
