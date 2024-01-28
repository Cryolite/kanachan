#define PY_SSIZE_T_CLEAN
#include "simulation/decision_maker.hpp"

#include "simulation/game_log.hpp"
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
#include <iostream>
#include <iomanip>
#include <vector>
#include <functional>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
        python::object device, python::object dtype, python::object baseline_model,
        python::list baseline_keys_to_be_deleted, python::object proposed_model,
        python::list proposed_keys_to_be_deleted, std::size_t batch_size);

    Impl_(Impl_ const &) = delete;

    Impl_ &operator=(Impl_ const &) = delete;

    void shrinkBatchSizeToFitNumThreads(std::size_t new_threads);

private:
    std::tuple<python::object, python::object> decide_(
        python::object sparse_batch, python::object numeric_batch,
        python::object progression_batch, python::object candidates_batch, bool proposed);

    void threadMain_(std::stop_token stop_token);

public:
    std::uint_fast16_t operator()(
        bool proposed, std::uint_fast8_t seat, std::vector<std::uint_fast16_t> &&sparse,
        std::vector<std::uint_fast32_t> &&numeric, std::vector<std::uint_fast16_t> &&progression,
        std::vector<std::uint_fast16_t> &&candidates, std::stop_token stop_token,
        Kanachan::GameLog &game_log);

    void run();

private:
    python::object torch_;
    python::object tensordict_;
    python::object tensor_;
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
    python::object device_;
    python::object dtype_;
    std::size_t batch_size_;
    python::object baseline_model_;
    std::vector<std::uint_fast8_t> baseline_seat_batch_;
    std::vector<std::vector<std::uint_fast16_t>> baseline_sparse_batch_;
    std::vector<std::vector<std::uint_fast32_t>> baseline_numeric_batch_;
    std::vector<std::vector<std::uint_fast16_t>> baseline_progression_batch_;
    std::vector<std::vector<std::uint_fast16_t>> baseline_candidates_batch_;
    std::vector<long> baseline_decision_batch_;
    std::vector<Kanachan::GameLog *> baseline_game_logs_;
    std::size_t baseline_batch_index_;
    std::size_t baseline_result_count_;
    python::list baseline_keys_to_be_deleted_;
    python::object proposed_model_;
    std::vector<std::uint_fast8_t> proposed_seat_batch_;
    std::vector<std::vector<std::uint_fast16_t>> proposed_sparse_batch_;
    std::vector<std::vector<std::uint_fast32_t>> proposed_numeric_batch_;
    std::vector<std::vector<std::uint_fast16_t>> proposed_progression_batch_;
    std::vector<std::vector<std::uint_fast16_t>> proposed_candidates_batch_;
    std::vector<long> proposed_decision_batch_;
    std::vector<Kanachan::GameLog *> proposed_game_logs_;
    std::size_t proposed_batch_index_;
    std::size_t proposed_result_count_;
    python::list proposed_keys_to_be_deleted_;
    std::uint_fast8_t state_;
    mutable std::mutex mtx_;
    std::condition_variable_any cv_;
}; // class DecisionMaker::Impl_

DecisionMaker::DecisionMaker(
    python::object device, python::object dtype, python::object baseline_model, 
    python::list baseline_keys_to_be_deleted, python::object proposed_model,
    python::list proposed_keys_to_be_deleted, std::size_t const batch_size)
    : p_impl_()
{
    if (PyGILState_Check() == 0) {
        KANACHAN_THROW<std::runtime_error>("The Python GIL must be held.");
    }
    if (device.is_none()) {
        KANACHAN_THROW<std::invalid_argument>("`device` must not be `None`.");
    }
    if (dtype.is_none()) {
        KANACHAN_THROW<std::invalid_argument>("`dtype` must not be `None`.");
    }
    if (baseline_model.is_none()) {
        KANACHAN_THROW<std::invalid_argument>("`baseline_model` must not be `None`.");
    }
    if (proposed_model.is_none()) {
        KANACHAN_THROW<std::invalid_argument>("`proposed_model` must not be `None`.");
    }
    if (batch_size == 0u) {
        KANACHAN_THROW<std::invalid_argument>("`batch_size` must be a positive integer.");
    }

    p_impl_ = std::make_shared<Impl_>(
        device, dtype, baseline_model, baseline_keys_to_be_deleted, proposed_model,
        proposed_keys_to_be_deleted, batch_size);
}

void DecisionMaker::shrinkBatchSizeToFitNumThreads(std::size_t const num_threads)
{
    if (PyGILState_Check() == 1) {
        KANACHAN_THROW<std::runtime_error>("The Python GIL must not be held.");
    }
    KANACHAN_ASSERT((!!p_impl_));
    p_impl_->shrinkBatchSizeToFitNumThreads(num_threads);
}

std::uint_fast16_t DecisionMaker::operator()(
    bool const proposed, std::uint8_t const seat, std::vector<std::uint_fast16_t> &&sparse,
    std::vector<std::uint_fast32_t> &&numeric, std::vector<std::uint_fast16_t> &&progression,
    std::vector<std::uint_fast16_t> &&candidates, std::stop_token stop_token,
    Kanachan::GameLog &game_log)
{
    if (PyGILState_Check() == 1) {
        KANACHAN_THROW<std::runtime_error>("The Python GIL must not be held.");
    }
    KANACHAN_ASSERT((!!p_impl_));
    return (*p_impl_)(
        proposed, seat, std::move(sparse), std::move(numeric), std::move(progression),
        std::move(candidates), stop_token, game_log);
}

void DecisionMaker::run()
{
    KANACHAN_ASSERT((!!p_impl_));
    p_impl_->run();
}

DecisionMaker::Impl_::Impl_(
    python::object device, python::object dtype, python::object baseline_model,
    python::list baseline_keys_to_be_deleted, python::object proposed_model,
    python::list proposed_keys_to_be_deleted, std::size_t const batch_size)
try : torch_(python::import("torch"))
    , tensordict_(python::import("tensordict"))
    , tensor_(torch_.attr("tensor"))
    , long_tensor_kwargs_()
    , long_zeros_()
    , long_tensor_constructor_()
    , constants_(python::import("kanachan.constants"))
    , num_types_of_sparse_features_(constants_.attr("NUM_TYPES_OF_SPARSE_FEATURES"))
    , max_num_active_sparse_features_(constants_.attr("MAX_NUM_ACTIVE_SPARSE_FEATURES"))
    , num_numeric_features_(constants_.attr("NUM_NUMERIC_FEATURES"))
    , num_types_of_progression_features_(constants_.attr("NUM_TYPES_OF_PROGRESSION_FEATURES"))
    , max_length_of_progression_features_(constants_.attr("MAX_LENGTH_OF_PROGRESSION_FEATURES"))
    , num_types_of_actions_(constants_.attr("NUM_TYPES_OF_ACTIONS"))
    , max_num_action_candidates_(constants_.attr("MAX_NUM_ACTION_CANDIDATES"))
    , device_(device)
    , dtype_(dtype)
    , batch_size_(batch_size)
    , baseline_model_(baseline_model)
    , baseline_seat_batch_(batch_size_)
    , baseline_sparse_batch_(batch_size_)
    , baseline_numeric_batch_(batch_size_)
    , baseline_progression_batch_(batch_size_)
    , baseline_candidates_batch_(batch_size_)
    , baseline_decision_batch_()
    , baseline_game_logs_(batch_size, nullptr)
    , baseline_batch_index_(0u)
    , baseline_result_count_(0u)
    , baseline_keys_to_be_deleted_(baseline_keys_to_be_deleted)
    , proposed_model_(proposed_model)
    , proposed_seat_batch_(batch_size)
    , proposed_sparse_batch_(batch_size_)
    , proposed_numeric_batch_(batch_size_)
    , proposed_progression_batch_(batch_size_)
    , proposed_candidates_batch_(batch_size_)
    , proposed_decision_batch_(batch_size_)
    , proposed_game_logs_(batch_size, nullptr)
    , proposed_batch_index_(0u)
    , proposed_result_count_(0u)
    , proposed_keys_to_be_deleted_(proposed_keys_to_be_deleted)
    , state_(0u)
    , mtx_()
    , cv_()
{
    KANACHAN_ASSERT((!device_.is_none()));
    KANACHAN_ASSERT((!dtype_.is_none()));
    KANACHAN_ASSERT((!baseline_model_.is_none()));
    KANACHAN_ASSERT((!proposed_model_.is_none()));
    KANACHAN_ASSERT((batch_size_ != 0u));

    long_tensor_kwargs_["device"] = device_;
    long_tensor_kwargs_["dtype"] = torch_.attr("int32");
    long_zeros_ = [this](python::tuple args) {
        return this->torch_.attr("zeros")(*args, **this->long_tensor_kwargs_);
    };
    long_tensor_constructor_ = [this](python::tuple args) {
        return this->tensor_(*args, **this->long_tensor_kwargs_);
    };
}
catch (python::error_already_set const &) {
    Kanachan::translatePythonException();
}

void DecisionMaker::Impl_::shrinkBatchSizeToFitNumThreads(std::size_t const num_threads)
{
    KANACHAN_ASSERT((PyGILState_Check() == 0));

    std::unique_lock lock(mtx_);

    if (num_threads == 0u) {
        if (baseline_batch_index_ > 0u) {
            KANACHAN_THROW<std::runtime_error>(_1) << baseline_batch_index_;
        }
        if (baseline_result_count_ > 0u) {
            KANACHAN_THROW<std::runtime_error>(_1) << baseline_result_count_;
        }
        if (proposed_batch_index_ > 0u) {
            KANACHAN_THROW<std::runtime_error>(_1) << proposed_batch_index_;
        }
        if (proposed_result_count_ > 0u) {
            KANACHAN_THROW<std::runtime_error>(_1) << proposed_result_count_;
        }

        state_ = 3u;
        cv_.notify_all();

        return;
    }

    if (num_threads >= batch_size_ * 2u - 1u) {
        return;
    }

    cv_.wait(lock, [&]{ return state_ == 0u; });
    std::size_t const log2_new_batch_size = std::log2(num_threads + 1u) - 1u;
    std::size_t const new_batch_size = 1 << log2_new_batch_size;
    batch_size_ = std::min(batch_size_, new_batch_size);
    if (num_threads < batch_size_ * 2u - 1u) {
        KANACHAN_THROW<std::logic_error>(_1)
            << "num_threads == " << num_threads << ", batch_size_ == " << batch_size_;
    }
    if (baseline_batch_index_ >= batch_size_) {
        if (baseline_result_count_ > 0u) {
            KANACHAN_THROW<std::logic_error>(_1)
                << "baseline_result_count_ == " << baseline_result_count_;
        }
        state_ = 1u;
        cv_.notify_all();
    }
    if (proposed_batch_index_ >= batch_size_) {
        if (proposed_result_count_ > 0u) {
            KANACHAN_THROW<std::logic_error>(_1)
                << "proposed_result_count_ == " << proposed_result_count_;
        }
        state_ = 1u;
        cv_.notify_all();
    }
}

std::tuple<python::object, python::object> DecisionMaker::Impl_::decide_(
    python::object sparse_batch, python::object numeric_batch, python::object progression_batch,
    python::object candidates_batch, bool const proposed)
try {
    KANACHAN_ASSERT((PyGILState_Check() == 1));

    long const batch_size = python::extract<long>(sparse_batch.attr("size")(0))();

    // data = TensorDict({
    //     'sparse': sparse,
    //     'numeric': numeric,
    //     'progression': progression,
    //     'candidates': candidates }, batch_size=[batch_size], device=device_)
    python::object data;
    {
        python::dict source;
        source["sparse"] = sparse_batch;
        source["numeric"] = numeric_batch;
        source["progression"] = progression_batch;
        source["candidates"] = candidates_batch;
        python::tuple args = python::make_tuple(source);
        python::dict kwargs;
        kwargs["batch_size"] = python::list();
        kwargs["batch_size"].attr("append")(batch_size);
        kwargs["device"] = device_;
        data = tensordict_.attr("TensorDict")(*args, **kwargs);
    }

    {
        python::object model = proposed ? proposed_model_ : baseline_model_;
        model(data);

        python::list keys_to_be_deleted = proposed ? proposed_keys_to_be_deleted_ : baseline_keys_to_be_deleted_;
        for (python::ssize_t i = 0u; i < python::len(keys_to_be_deleted); ++i) {
            python::object key = keys_to_be_deleted[i];
            if (!data.attr("get")(key, python::object()).is_none()) {
                data.attr("del_")(key);
            }
        }
        // The `detach_` method cannot be used in conjunction with DDP.
        data = data.attr("detach")();
    }

    // action_index_batch = data.get('action', None)
    python::object action_index_batch = data.attr("get")("action", python::object());
    if (action_index_batch.is_none()) {
        KANACHAN_THROW<std::runtime_error>("The model did not output the `action` tensor.");
    }
    if (action_index_batch.attr("dim")() != 1) {
        long const dim = python::extract<long>(action_index_batch.attr("dim")())();
        KANACHAN_THROW<std::runtime_error>(_1) << dim << ": An invalid dimension.";
    }
    if (action_index_batch.attr("size")(0) != batch_size) {
        long const size = python::extract<long>(action_index_batch.attr("size")(0))();
        KANACHAN_THROW<std::runtime_error>(_1) << size << " != " << batch_size;
    }

    // log_prob_batch = data.get('sample_log_prob', None)
    python::object log_prob_batch = data.attr("get")("sample_log_prob", python::object());
    if (!log_prob_batch.is_none()) {
        if (log_prob_batch.attr("dim")() != 1) {
            long const dim = python::extract<long>(log_prob_batch.attr("dim")())();
            KANACHAN_THROW<std::runtime_error>(_1) << dim << ": An invalid dimension.";
        }
        if (log_prob_batch.attr("size")(0) != batch_size) {
            long const size = python::extract<long>(log_prob_batch.attr("size")(0))();
            KANACHAN_THROW<std::runtime_error>(_1) << size << " != " << batch_size;
        }
    }

    for (python::ssize_t i = 0u; i < batch_size; ++i) {
        python::object index = action_index_batch[i].attr("item")();
        python::object candidate = candidates_batch[i][index];
        if (candidate >= num_types_of_actions_) {
            std::ostringstream oss;
            oss << "An invalid decision: " << python::extract<long>(index)() << '\n';
            for (python::ssize_t j = 0u; j < max_num_action_candidates_; ++j) {
                long const candidate_ = python::extract<long>(candidates_batch[i][j].attr("item")());
                oss << "  " << candidate_  << '\n';
            }
            KANACHAN_THROW<std::logic_error>(_1) << oss.str();
        }
    }

    python::object decision_batch;
    {
        python::object arange = torch_.attr("arange")(batch_size);
        decision_batch \
            = candidates_batch.attr("__getitem__")(python::make_tuple(arange, action_index_batch));
    }

    return { data, decision_batch };
}
catch (python::error_already_set const &) {
    try {
        Kanachan::translatePythonException();
    }
    catch (std::runtime_error const &e) {
        std::cerr << e.what() << std::endl;
        std::abort();
    }
}
catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
    std::abort();
}

void DecisionMaker::Impl_::run()
try {
    for (;;) {
        bool proposed = false;
        std::size_t batch_index = SIZE_MAX;
        std::vector<std::uint_fast8_t> seat_batch;
        std::vector<std::vector<std::uint_fast16_t>> sparse_batch;
        std::vector<std::vector<std::uint_fast32_t>> numeric_batch;
        std::vector<std::vector<std::uint_fast16_t>> progression_batch;
        std::vector<std::vector<std::uint_fast16_t>> candidates_batch;
        std::vector<Kanachan::GameLog *> game_logs;
        {
            Kanachan::GIL::RecursiveRelease gil_release;
            std::unique_lock lock(mtx_);

            cv_.wait(lock, [this]() { return state_ == 1u || state_ == 3u; });
            if (state_ == 3u) {
                if (baseline_batch_index_ > 0u) {
                    KANACHAN_THROW<std::logic_error>(_1) << baseline_batch_index_;
                }
                if (baseline_result_count_ > 0u) {
                    KANACHAN_THROW<std::logic_error>(_1) << baseline_result_count_;
                }
                if (proposed_batch_index_ > 0u) {
                    KANACHAN_THROW<std::logic_error>(_1) << proposed_batch_index_;
                }
                if (proposed_result_count_ > 0u) {
                    KANACHAN_THROW<std::logic_error>(_1) << proposed_result_count_;
                }
                break;
            }
            if (baseline_batch_index_ >= batch_size_) {
                proposed = false;
            }
            else {
                if (proposed_batch_index_ < batch_size_) {
                    KANACHAN_THROW<std::logic_error>(_1)
                        << "baseline_batch_index_ == " << baseline_batch_index_ << ", "
                        << "proposed_batch_index_ == " << proposed_batch_index_ << ", "
                        << "batch_size_ == " << batch_size_;
                }
                proposed = true;
            }
            if (baseline_result_count_ > 0u) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << "baseline_result_count_ == " << baseline_result_count_;
            }
            if (proposed_result_count_ > 0u) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << "proposed_result_count_ == " << proposed_result_count_;
            }

            batch_index = proposed ? proposed_batch_index_ : baseline_batch_index_;
            auto &seat_batch_ = proposed ? proposed_seat_batch_ : baseline_seat_batch_;
            auto &sparse_batch_ = proposed ? proposed_sparse_batch_ : baseline_sparse_batch_;
            auto &numeric_batch_ = proposed ? proposed_numeric_batch_ : baseline_numeric_batch_;
            auto &progression_batch_ = proposed ? proposed_progression_batch_ : baseline_progression_batch_;
            auto &candidates_batch_ = proposed ? proposed_candidates_batch_ : baseline_candidates_batch_;
            auto &game_logs_ = proposed ? proposed_game_logs_ : baseline_game_logs_;

            if (seat_batch_.size() < batch_index) {
                KANACHAN_THROW<std::logic_error>(_1) << seat_batch_.size() << " < " << batch_index;
            }
            seat_batch_.resize(batch_index);

            if (sparse_batch_.size() < batch_index) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << sparse_batch_.size() << " < " << batch_index;
            }
            sparse_batch_.resize(batch_index);

            if (numeric_batch_.size() < batch_index) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << numeric_batch_.size() << " < " << batch_index;
            }
            numeric_batch_.resize(batch_index);

            if (progression_batch_.size() < batch_index) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << progression_batch_.size() << " < " << batch_index;
            }
            progression_batch_.resize(batch_index);

            if (candidates_batch_.size() < batch_index) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << candidates_batch_.size() << " < " << batch_index;
            }
            candidates_batch_.resize(batch_index);

            if (game_logs_.size() < batch_index) {
                KANACHAN_THROW<std::logic_error>(_1) << game_logs_.size() << " < " << batch_index;
            }
            game_logs_.resize(batch_index);

            seat_batch = seat_batch_;
            sparse_batch = sparse_batch_;
            numeric_batch = numeric_batch_;
            progression_batch = progression_batch_;
            candidates_batch = candidates_batch_;
            game_logs = game_logs_;
        }

        std::vector<long> decision_batch;
        {
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
                        numeric_.append(v);
                    }
                    numeric_batch_tmp.append(numeric_);
                }
                return long_tensor_constructor_(python::make_tuple(numeric_batch_tmp));
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
                    while (python::len(candidates_) < max_num_action_candidates_) {
                        // Padding.
                        candidates_.append(num_types_of_actions_);
                    }
                    candidates_batch_tmp.append(candidates_);
                }
                return long_tensor_constructor_(python::make_tuple(candidates_batch_tmp));
            }();

            auto [data, decision_batch_tmp] = decide_(
                sparse_batch_tmp, numeric_batch_tmp, progression_batch_tmp, candidates_batch_tmp,
                proposed);

            for (std::size_t i = 0u; i < batch_index; ++i) {
                std::uint_fast8_t const seat = seat_batch[i];
                game_logs[i]->onDecision(
                    seat, data[i].attr("cpu")().attr("detach")().attr("clone")());
            }

            for (std::size_t i = 0u; i < batch_index; ++i) {
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
            Kanachan::GIL::RecursiveRelease gil_release;
            std::unique_lock lock(mtx_);
            if (proposed) {
                proposed_decision_batch_.swap(decision_batch);
                proposed_result_count_ = batch_index;
                proposed_batch_index_ = 0u;
            }
            else {
                baseline_decision_batch_.swap(decision_batch);
                baseline_result_count_ = batch_index;
                baseline_batch_index_ = 0u;
            }
            state_ = 2u;
        }

        cv_.notify_all();
    }
}
catch (python::error_already_set const &) {
    try {
        Kanachan::translatePythonException();
    }
    catch (std::runtime_error const &e) {
        std::cerr << e.what() << std::endl;
        std::abort();
    }
}

std::uint_fast16_t DecisionMaker::Impl_::operator()(
    bool const proposed, std::uint_fast8_t const seat, std::vector<std::uint_fast16_t> &&sparse,
    std::vector<std::uint_fast32_t> &&numeric, std::vector<std::uint_fast16_t> &&progression,
    std::vector<std::uint_fast16_t> &&candidates, std::stop_token stop_token,
    Kanachan::GameLog &game_log)
try {
    KANACHAN_ASSERT((PyGILState_Check() == 0));

    if (candidates.size() == 0u) {
        KANACHAN_THROW<std::invalid_argument>("`candidates` is empty.");
    }

    std::uint_fast16_t decision = UINT_FAST16_MAX;
    {
        std::unique_lock lock(mtx_);

        cv_.wait(lock, stop_token, [this]() { return state_ == 0; });
        if (stop_token.stop_requested()) {
            KANACHAN_THROW<Kanachan::ThreadTermination>("Graceful termination.");
        }

        std::size_t &batch_index = proposed ? proposed_batch_index_ : baseline_batch_index_;

        std::size_t &result_count = proposed ? proposed_result_count_ : baseline_result_count_;
        if (result_count > 0u) {
            KANACHAN_THROW<std::logic_error>(_1) << "result_count == " << result_count;
        }

        auto &seat_batch = proposed ? proposed_seat_batch_ : baseline_seat_batch_;
        auto &sparse_batch = proposed ? proposed_sparse_batch_ : baseline_sparse_batch_;
        auto &numeric_batch = proposed ? proposed_numeric_batch_ : baseline_numeric_batch_;
        auto &progression_batch = proposed ? proposed_progression_batch_ : baseline_progression_batch_;
        auto &candidates_batch = proposed ? proposed_candidates_batch_ : baseline_candidates_batch_;
        auto &game_logs = proposed ? proposed_game_logs_ : baseline_game_logs_;

        seat_batch[batch_index] = seat;
        sparse_batch[batch_index].swap(sparse);
        numeric_batch[batch_index].swap(numeric);
        progression_batch[batch_index].swap(progression);
        candidates_batch[batch_index].swap(candidates);
        game_logs[batch_index] = &game_log;

        std::size_t const my_index = batch_index++;
        if (batch_index == batch_size_) {
            state_ = 1u;
            cv_.notify_all();
        }

        cv_.wait(lock, stop_token, [&]() { return state_ == 2u && batch_index == 0u; });
        if (stop_token.stop_requested()) {
            KANACHAN_THROW<Kanachan::ThreadTermination>("Graceful termination.");
        }
        if (result_count == 0u) {
            KANACHAN_THROW<std::logic_error>(_1) << "result_count == " << result_count;
        }

        decision = proposed ? proposed_decision_batch_[my_index] : baseline_decision_batch_[my_index];

        --result_count;
        if (result_count == 0u) {
            if (seat_batch.size() < batch_size_) {
                KANACHAN_THROW<std::logic_error>(_1) << seat_batch.size() << " < " << batch_size_;
            }
            seat_batch.resize(batch_size_);

            if (sparse_batch.size() < batch_size_) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << sparse_batch.size() << " < " << batch_size_;
            }
            sparse_batch.resize(batch_size_);

            if (numeric_batch.size() < batch_size_) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << numeric_batch.size() << " < " << batch_size_;
            }
            numeric_batch.resize(batch_size_);

            if (progression_batch.size() < batch_size_) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << progression_batch.size() << " < " << batch_size_;
            }
            progression_batch.resize(batch_size_);

            if (candidates_batch.size() < batch_size_) {
                KANACHAN_THROW<std::logic_error>(_1)
                    << candidates_batch.size() << " < " << batch_size_;
            }
            candidates_batch.resize(batch_size_);

            if (game_logs.size() < batch_size_) {
                KANACHAN_THROW<std::logic_error>(_1) << game_logs.size() << " < " << batch_size_;
            }
            game_logs.resize(batch_size_);

            if (baseline_batch_index_ >= batch_size_ || proposed_batch_index_ >= batch_size_) {
                state_ = 1u;
            }
            else {
                state_ = 0u;
            }
            cv_.notify_all();
        }
    }

    return decision;
}
catch (python::error_already_set const &) {
    Kanachan::translatePythonException();
}

} // namespace Kanachan
