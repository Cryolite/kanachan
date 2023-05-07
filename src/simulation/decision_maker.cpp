#define PY_SSIZE_T_CLEAN
#include "simulation/decision_maker.hpp"

#include "simulation/utility.hpp"
#include "simulation/gil.hpp"
#include "common/thread.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/import.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/str.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/long.hpp>
#include <boost/python/object.hpp>
#include <boost/python/errors.hpp>
#include <Python.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stop_token>
#include <sstream>
#include <iomanip>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <cstddef>


namespace {

namespace python = boost::python;
using std::placeholders::_1;

} // namespace `anonymous`

namespace Kanachan{

class DecisionMaker::Impl_
{
public:
    Impl_(
        std::string const &device, python::object dtype, python::object model,
        std::size_t batch_size);

    Impl_(Impl_ const &) = delete;

    Impl_ &operator=(Impl_ const &) = delete;

    void shrinkBatchSizeToFitNumThreads(std::size_t new_threads);

private:
    python::object decide_(
        python::object sparse_batch, python::object numeric_batch,
        python::object progression_batch, python::object candidates_batch);

    void threadMain_(std::stop_token stop_token);

public:
    std::uint_fast16_t operator()(
        std::vector<std::uint_fast16_t> &&sparse, std::vector<std::uint_fast32_t> &&numeric,
        std::vector<std::uint_fast16_t> &&progression, std::vector<std::uint_fast16_t> &&candidates,
        std::stop_token stop_token);

    void join();

private:
    python::object torch_;
    python::object tensor_;
    python::dict dtype_tensor_kwargs_;
    std::function<python::object(python::tuple)> dtype_zeros_;
    std::function<python::object(python::tuple)> dtype_tensor_constructor_;
    python::dict long_tensor_kwargs_;
    std::function<python::object(python::tuple)> long_zeros_;
    std::function<python::object(python::tuple)> long_tensor_constructor_;
    python::object constants_;
    python::object num_types_of_sparse_features_;
    python::object max_num_active_sparse_features_;
    python::object num_numeric_features_;
    python::object num_types_of_progression_features_;
    python::object max_length_of_progression_features_;
    python::object num_types_of_actions_;
    python::object max_num_action_candidates_;
    python::str device_;
    python::object dtype_;
    python::object model_;
    std::jthread thread_;
    std::size_t batch_size_;
    std::vector<std::vector<std::uint_fast16_t>> sparse_batch_;
    std::vector<std::vector<std::uint_fast32_t>> numeric_batch_;
    std::vector<std::vector<std::uint_fast16_t>> progression_batch_;
    std::vector<std::vector<std::uint_fast16_t>> candidates_batch_;
    std::vector<long> decision_batch_;
    std::size_t batch_index_;
    std::size_t result_count_;
    mutable std::mutex mtx_;
    std::condition_variable_any ready_to_enqueue_;
    std::condition_variable_any ready_to_run_;
    std::condition_variable_any ready_to_dequeue_;
}; // class DecisionMaker::Impl_

DecisionMaker::DecisionMaker(
    std::string const &device, python::object dtype, python::object model, std::size_t batch_size)
    : p_impl_()
{
    if (PyGILState_Check() != 1) {
        KANACHAN_THROW<std::runtime_error>("The Python GIL must be held.");
    }
    if (dtype.is_none()) {
        KANACHAN_THROW<std::invalid_argument>("`dtype` must not be `None`.");
    }
    if (model.is_none()) {
        KANACHAN_THROW<std::invalid_argument>("`model` must not be `None`.");
    }
    if (batch_size == 0u) {
        KANACHAN_THROW<std::invalid_argument>("`batch_size` must be a positive integer.");
    }

    p_impl_ = std::make_shared<Impl_>(device, dtype, model, batch_size);
}

void DecisionMaker::shrinkBatchSizeToFitNumThreads(std::size_t const num_threads)
{
    if (PyGILState_Check() != 0) {
        KANACHAN_THROW<std::runtime_error>("The Python GIL must not be held.");
    }
    KANACHAN_ASSERT((!!p_impl_));
    p_impl_->shrinkBatchSizeToFitNumThreads(num_threads);
}

std::uint_fast16_t DecisionMaker::operator()(
    std::vector<std::uint_fast16_t> &&sparse, std::vector<std::uint_fast32_t> &&numeric,
    std::vector<std::uint_fast16_t> &&progression, std::vector<std::uint_fast16_t> &&candidates,
    std::stop_token stop_token)
{
    if (PyGILState_Check() != 0) {
        KANACHAN_THROW<std::runtime_error>("The Python GIL must not be held.");
    }
    KANACHAN_ASSERT((!!p_impl_));
    return (*p_impl_)(
        std::move(sparse), std::move(numeric), std::move(progression), std::move(candidates),
        stop_token);
}

void DecisionMaker::join()
{
    if (PyGILState_Check() != 0) {
        KANACHAN_THROW<std::runtime_error>("The Python GIL must not be held.");
    }
    KANACHAN_ASSERT((!!p_impl_));
    p_impl_->join();
}

DecisionMaker::Impl_::Impl_(
    std::string const &device, python::object dtype, python::object model, std::size_t batch_size)
try : torch_(python::import("torch"))
    , tensor_(torch_.attr("tensor"))
    , dtype_tensor_kwargs_()
    , dtype_zeros_()
    , dtype_tensor_constructor_()
    , long_tensor_kwargs_()
    , long_zeros_()
    , long_tensor_constructor_()
    , constants_(python::import("kanachan.training.constants"))
    , num_types_of_sparse_features_(constants_.attr("NUM_TYPES_OF_SPARSE_FEATURES"))
    , max_num_active_sparse_features_(constants_.attr("MAX_NUM_ACTIVE_SPARSE_FEATURES"))
    , num_numeric_features_(constants_.attr("NUM_NUMERIC_FEATURES"))
    , num_types_of_progression_features_(constants_.attr("NUM_TYPES_OF_PROGRESSION_FEATURES"))
    , max_length_of_progression_features_(constants_.attr("MAX_LENGTH_OF_PROGRESSION_FEATURES"))
    , num_types_of_actions_(constants_.attr("NUM_TYPES_OF_ACTIONS"))
    , max_num_action_candidates_(constants_.attr("MAX_NUM_ACTION_CANDIDATES"))
    , device_(device)
    , dtype_(dtype)
    , model_(model)
    , thread_()
    , batch_size_(batch_size)
    , sparse_batch_(batch_size_)
    , numeric_batch_(batch_size_)
    , progression_batch_(batch_size_)
    , candidates_batch_(batch_size_)
    , decision_batch_()
    , batch_index_(0u)
    , result_count_(0u)
    , mtx_()
    , ready_to_enqueue_()
    , ready_to_run_()
    , ready_to_dequeue_()
{
    KANACHAN_ASSERT((!device_.is_none()));
    KANACHAN_ASSERT((!dtype_.is_none()));
    KANACHAN_ASSERT((!model_.is_none()));
    KANACHAN_ASSERT((batch_size_ != 0u));

    dtype_tensor_kwargs_["device"] = device_;
    dtype_tensor_kwargs_["dtype"] = dtype_;
    dtype_zeros_ = [this](python::tuple args) {
        return this->torch_.attr("zeros")(*args, **this->dtype_tensor_kwargs_);
    };
    dtype_tensor_constructor_ = [this](python::tuple args) {
        return this->tensor_(*args, **this->dtype_tensor_kwargs_);
    };

    long_tensor_kwargs_["device"] = device_;
    long_tensor_kwargs_["dtype"] = torch_.attr("long");
    long_zeros_ = [this](python::tuple args) {
        return this->torch_.attr("zeros")(*args, **this->long_tensor_kwargs_);
    };
    long_tensor_constructor_ = [this](python::tuple args) {
        return this->tensor_(*args, **this->long_tensor_kwargs_);
    };

    thread_ = std::jthread(&Impl_::threadMain_, this);
}
catch (python::error_already_set const &) {
    Kanachan::translatePythonException();
}

void DecisionMaker::Impl_::shrinkBatchSizeToFitNumThreads(std::size_t const num_threads)
{
    KANACHAN_ASSERT((PyGILState_Check() == 0));

    std::unique_lock lock(mtx_);

    if (num_threads == 0u) {
        if (batch_index_ > 0u) {
            KANACHAN_THROW<std::runtime_error>(_1) << batch_index_;
        }
        if (result_count_ > 0u) {
            KANACHAN_THROW<std::runtime_error>(_1) << result_count_;
        }
        return;
    }

    if (num_threads >= batch_size_ * 2u - 1u) {
        return;
    }

    std::size_t const log2_new_batch_size = std::log2(num_threads + 1u) - 1.0;
    std::size_t const new_batch_size = 1 << log2_new_batch_size;
    batch_size_ = new_batch_size;
    if (batch_index_ >= batch_size_ && result_count_ == 0u) {
        ready_to_run_.notify_all();
    }
}

python::object DecisionMaker::Impl_::decide_(
    python::object sparse_batch, python::object numeric_batch, python::object progression_batch,
    python::object candidates_batch)
try {
    KANACHAN_ASSERT((batch_index_ >= 1u));
    KANACHAN_ASSERT((result_count_ == 0u));

    python::object weight_batch = model_(
        sparse_batch, numeric_batch, progression_batch, candidates_batch);
    if (weight_batch.attr("dim")() != 2) {
        long const dim = python::extract<long>(weight_batch.attr("dim")())();
        KANACHAN_THROW<std::runtime_error>(_1) << dim << ": An invalid dimension.";
    }
    if (weight_batch.attr("size")(0) != python::len(sparse_batch)) {
        long const size = python::extract<long>(weight_batch.attr("size")(0))();
        KANACHAN_THROW<std::runtime_error>(_1) << size << " != " << batch_size_;
    }
    if (weight_batch.attr("size")(1) != max_num_action_candidates_) {
        long const size = python::extract<long>(weight_batch.attr("size")(1))();
        KANACHAN_THROW<std::runtime_error>(_1)
            << size << " != " << python::extract<long>(max_num_action_candidates_)();
    }

    {
        python::object mask = (candidates_batch < num_types_of_actions_);
        weight_batch = torch_.attr("where")(
            mask, weight_batch, -std::numeric_limits<double>::infinity());
    }

    python::dict kwargs;
    kwargs["dim"] = python::long_(1);
    python::object index_batch = torch_.attr("argmax")(*python::make_tuple(weight_batch), **kwargs);
    for (python::ssize_t i = 0u; i < python::len(index_batch); ++i) {
        python::object index = index_batch[i].attr("item")();
        python::object candidate = candidates_batch[i][index];
        if (candidate >= num_types_of_actions_) {
            std::ostringstream oss;
            oss << "An invalid decision:\n";
            for (python::ssize_t j = 0u; j < python::len(candidates_batch[i]); ++j) {
                long candidate_ = python::extract<long>(candidates_batch[i][j].attr("item")());
                double weight_ = python::extract<double>(weight_batch[i][j].attr("item")());
                oss << std::setw(3) << candidate_ << ": " << weight_ << '\n';
            }
            KANACHAN_THROW<std::logic_error>(_1) << oss.str();
        }
    }

    python::ssize_t const original_batch_size = python::len(index_batch);
    python::object arange = torch_.attr("arange")(original_batch_size);
    return candidates_batch.attr("__getitem__")(python::make_tuple(arange, index_batch));
}
catch (python::error_already_set const &) {
    Kanachan::translatePythonException();
}

void DecisionMaker::Impl_::threadMain_(std::stop_token stop_token)
try {
    for (;;) {
        std::vector<std::vector<std::uint_fast16_t>> sparse_batch;
        std::vector<std::vector<std::uint_fast32_t>> numeric_batch;
        std::vector<std::vector<std::uint_fast16_t>> progression_batch;
        std::vector<std::vector<std::uint_fast16_t>> candidates_batch;
        {
            std::unique_lock lock(mtx_);
            ready_to_run_.wait(
                lock, stop_token,
                [this]() { return batch_index_ >= batch_size_ && result_count_ == 0u; });
            if (stop_token.stop_requested() && batch_index_ == 0u) {
                return;
            }
            ready_to_run_.wait(lock, [this]() { return result_count_ == 0u; });

            sparse_batch = sparse_batch_;
            numeric_batch = numeric_batch_;
            progression_batch = progression_batch_;
            candidates_batch = candidates_batch_;
        }

        std::vector<long> decision_batch;
        {
            Kanachan::GIL::RecursiveLock gil_lock;

            python::object sparse_batch_tmp = [&]() {
                python::list sparse_batch_tmp;
                for (std::vector<std::uint_fast16_t> const &sparse : sparse_batch) {
                    python::list sparse_;
                    for (long const v : sparse) {
                        sparse_.append(v);
                    }
                    while (python::len(sparse_) < max_num_active_sparse_features_) {
                        // Padding.
                        sparse_.append(num_types_of_sparse_features_);
                    }
                    sparse_batch_tmp.append(sparse_);
                }
                return long_tensor_constructor_(python::make_tuple(sparse_batch_tmp));
            }();

            python::object numeric_batch_tmp = [&]() {
                python::list numeric_batch_tmp;
                for (std::vector<std::uint_fast32_t> const &numeric : numeric_batch) {
                    python::list numeric_;
                    for (long const v : numeric) {
                        numeric_.append(static_cast<double>(v));
                    }
                    while (python::len(numeric_) < num_numeric_features_) {
                        // Padding.
                        numeric_.append(0.0);
                    }
                    for (long i = 2; i < num_numeric_features_; ++i) {
                        // Scaling.
                        numeric_[i] /= 10000.0;
                    }
                    numeric_batch_tmp.append(numeric_);
                }
                return dtype_tensor_constructor_(python::make_tuple(numeric_batch_tmp));
            }();

            python::object progression_batch_tmp = [&]() {
                python::list progression_batch_tmp;
                for (std::vector<std::uint_fast16_t> const &progression : progression_batch) {
                    python::list progression_;
                    for (long const v :progression) {
                        progression_.append(v);
                    }
                    while (python::len(progression_) < max_length_of_progression_features_) {
                        // Padding.
                        progression_.append(num_types_of_progression_features_);
                    }
                    progression_batch_tmp.append(progression_);
                }
                return long_tensor_constructor_(python::make_tuple(progression_batch_tmp));
            }();

            python::object candidates_batch_tmp = [&]() {
                python::list candidates_batch_tmp;
                for (std::vector<std::uint_fast16_t> const &candidates : candidates_batch) {
                    python::list candidates_;
                    for (long const v : candidates) {
                        candidates_.append(v);
                    }
                    // Append `<V>`.
                    candidates_.append(num_types_of_actions_);
                    while (python::len(candidates_) < max_num_action_candidates_) {
                        // Padding.
                        candidates_.append(num_types_of_actions_ + 1);
                    }
                    candidates_batch_tmp.append(candidates_);
                }
                return long_tensor_constructor_(python::make_tuple(candidates_batch_tmp));
            }();

            python::object decision_batch_tmp = decide_(
                sparse_batch_tmp, numeric_batch_tmp, progression_batch_tmp, candidates_batch_tmp);

            for (std::size_t i = 0u; i < batch_index_; ++i) {
                long const decision = [&]() {
                    python::object decision = decision_batch_tmp[i].attr("item")();
                    python::extract<long> decision_(decision);
                    if (!decision_.check()) {
                        KANACHAN_THROW<std::runtime_error>(_1)
                            << decision.attr("__class__").attr("__name__")
                            << ": An invalid type of `decision`.";
                    }
                    return decision_();
                }();
                decision_batch.push_back(decision);
            }
        }

        {
            std::unique_lock lock(mtx_);
            decision_batch_.swap(decision_batch);
            result_count_ = batch_index_;
            batch_index_ = 0u;
        }
        ready_to_dequeue_.notify_all();
    }
}
catch (python::error_already_set const &) {
    Kanachan::translatePythonException();
}

std::uint_fast16_t DecisionMaker::Impl_::operator()(
    std::vector<std::uint_fast16_t> &&sparse, std::vector<std::uint_fast32_t> &&numeric,
    std::vector<std::uint_fast16_t> &&progression, std::vector<std::uint_fast16_t> &&candidates,
    std::stop_token stop_token)
try {
    KANACHAN_ASSERT((PyGILState_Check() == 0));

    if (candidates.size() == 0u) {
        KANACHAN_THROW<std::invalid_argument>("An empty `candidates`.");
    }

    std::unique_lock lock(mtx_);
    ready_to_enqueue_.wait(
        lock, stop_token, [this]() { return batch_index_ < batch_size_ && result_count_ == 0u; });
    if (stop_token.stop_requested()) {
        KANACHAN_THROW<Kanachan::ThreadTermination>("Graceful termination.");
    }

    sparse_batch_[batch_index_].swap(sparse);
    numeric_batch_[batch_index_].swap(numeric);
    progression_batch_[batch_index_].swap(progression);
    candidates_batch_[batch_index_].swap(candidates);

    std::size_t const my_index = batch_index_++;
    if (batch_index_ == batch_size_) {
        ready_to_run_.notify_all();
    }

    ready_to_dequeue_.wait(lock, stop_token, [this]() { return result_count_ > 0u; });
    if (stop_token.stop_requested()) {
        KANACHAN_THROW<Kanachan::ThreadTermination>("Graceful termination.");
    }
    KANACHAN_ASSERT((batch_index_ == 0u));

    std::uint_fast16_t const decision = decision_batch_[my_index];
    --result_count_;
    if (result_count_ == 0u) {
        sparse_batch_.resize(batch_size_);
        numeric_batch_.resize(batch_size_);
        progression_batch_.resize(batch_size_);
        candidates_batch_.resize(batch_size_);
        if (batch_index_ < batch_size_) {
            ready_to_enqueue_.notify_all();
        }
        else {
            ready_to_run_.notify_all();
        }
    }

    return decision;
}
catch (python::error_already_set const &) {
    Kanachan::translatePythonException();
}

void DecisionMaker::Impl_::join()
{
    KANACHAN_ASSERT((PyGILState_Check() == 0));

    thread_.request_stop();
    thread_.join();
}

} // namespace Kanachan
