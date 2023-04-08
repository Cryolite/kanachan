#define PY_SSIZE_T_CLEAN
#include "simulation/gil.hpp"

#include "common/throw.hpp"
#include <Python.h>
#include <stdexcept>


namespace Kanachan::GIL{

RecursiveLock::RecursiveLock()
    : state_(PyGILState_Ensure())
    , owns_(true)
{}

RecursiveLock::RecursiveLock(RecursiveLock &&rhs) noexcept
    : state_(rhs.state_)
    , owns_(rhs.owns_)
{
    rhs.owns_ = false;
}

RecursiveLock::~RecursiveLock()
{
    if (owns_) {
        PyGILState_Release(state_);
    }
}

RecursiveLock &RecursiveLock::operator=(RecursiveLock &&rhs) noexcept
{
    state_ = rhs.state_;
    owns_ = rhs.owns_;
    rhs.owns_ = false;
    return *this;
}

RecursiveRelease::RecursiveRelease()
    : save_(nullptr)
{
    if (PyGILState_Check() != 0) {
        save_ = PyEval_SaveThread();
    }
}

RecursiveRelease::~RecursiveRelease()
{
    if (save_ != nullptr) {
        PyEval_RestoreThread(save_);
        save_ = nullptr;
    }
}

} // namespace Kanachan::GIL
