#include "simulation/simulator.hpp"

#include "simulation/game.hpp"
#include "simulation/game_log.hpp"
#include "simulation/paishan.hpp"
#include "simulation/decision_maker.hpp"
#include "simulation/utility.hpp"
#include "simulation/gil.hpp"
#include "common/thread.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/errors.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <Python.h>
#include <mutex>
#include <thread>
#include <stop_token>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <array>
#include <functional>
#include <utility>
#include <memory>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstdint>
#include <cstddef>


namespace{

using std::placeholders::_1;
namespace python = boost::python;

} // namespace `anonymous`

namespace Kanachan{

class Simulator::Impl_
{
private:
    using Grades_ = std::array<std::uint_fast8_t, 4u>;
    using SeatArrangement_ = std::array<bool, 4u>;
    using Seed_ = std::vector<std::uint_least32_t>;
    using GameInitializer_ = std::tuple<Grades_, SeatArrangement_, Seed_>;

public:
    Impl_(
        python::object device, python::object dtype, std::uint_fast8_t room,
        std::uint_fast8_t baseline_grade, python::object baseline_model,
        python::list baseline_keys_to_be_deleted, std::uint_fast8_t proposed_grade,
        python::object proposed_model, python::list proposed_keys_to_be_deleted,
        unsigned long simulation_mode, std::size_t num_simulation_sets, std::size_t batch_size,
        std::size_t concurrency, python::object progress);

    Impl_(Impl_ const &) = delete;

    Impl_ &operator=(Impl_ const &) = delete;

private:
    void threadMain_(std::stop_token stop_token);

public:
    python::list run();

private:
    bool dong_feng_zhan_;
    std::uint_fast8_t room_;
    std::shared_ptr<Kanachan::DecisionMaker> p_decision_maker_;
    std::vector<std::jthread> threads_;
    std::vector<GameInitializer_> game_initializers_;
    std::vector<std::shared_ptr<Kanachan::GameLog>> game_logs_;
    std::size_t num_alive_threads_;
    python::object progress_;
    std::mutex mtx_;
}; // class Simulator::Impl_

Simulator::Simulator(
    python::object device, python::object dtype, std::uint_fast8_t const room,
    std::uint_fast8_t const baseline_grade, python::object baseline_model,
    python::list baseline_keys_to_be_deleted, std::uint_fast8_t const proposed_grade,
    python::object proposed_model, python::list proposed_keys_to_be_deleted,
    unsigned long const simulation_mode, std::size_t const num_simulation_sets,
    std::size_t const batch_size, std::size_t const concurrency, python::object progress)
    : p_impl_(std::make_shared<Impl_>(
        device, dtype, room, baseline_grade, baseline_model, baseline_keys_to_be_deleted,
        proposed_grade, proposed_model, proposed_keys_to_be_deleted, simulation_mode,
        num_simulation_sets, batch_size, concurrency, progress))
{}

python::list Simulator::run()
{
    if (PyGILState_Check() == 0) {
        KANACHAN_THROW<std::runtime_error>("GIL must be held.");
    }
    KANACHAN_ASSERT((!!p_impl_));
    return p_impl_->run();
}

Simulator::Impl_::Impl_(
    python::object device, python::object dtype, std::uint_fast8_t const room,
    std::uint_fast8_t const baseline_grade, python::object baseline_model,
    python::list baseline_keys_to_be_deleted, std::uint_fast8_t const proposed_grade,
    python::object proposed_model, python::list proposed_keys_to_be_deleted,
    unsigned long const simulation_mode, std::size_t const num_simulation_sets,
    std::size_t const batch_size, std::size_t const concurrency, python::object progress)
    : dong_feng_zhan_(simulation_mode & 2u)
    , room_(room)
    , p_decision_maker_(
          std::make_shared<Kanachan::DecisionMaker>(
              device, dtype, baseline_model, baseline_keys_to_be_deleted, proposed_model,
              proposed_keys_to_be_deleted, batch_size))
    , threads_()
    , game_initializers_()
    , game_logs_()
    , num_alive_threads_(concurrency)
    , progress_(progress)
    , mtx_()
{
    if (num_simulation_sets == 0u) {
        KANACHAN_THROW<std::invalid_argument>("`num_simulation_sets` must be a positive integer.");
    }

    bool const no_duplicate = (simulation_mode & 1u);
    bool const one_versus_three = (simulation_mode & 4u);

    if (baseline_grade < 0 || 16 <= baseline_grade) {
        KANACHAN_THROW<std::invalid_argument>(_1)
            << baseline_grade << ": An invalid baseline grade.";
    }

    if (proposed_grade < 0 || 16 <= proposed_grade) {
        KANACHAN_THROW<std::invalid_argument>(_1)
            << proposed_grade << ": An invalid proposed grade.";
    }

    if (concurrency == 0u) {
        KANACHAN_THROW<std::invalid_argument>("`concurrency` must be a positive integer.");
    }
    if (concurrency < batch_size * 2u - 1u) {
        KANACHAN_THROW<std::invalid_argument>(_1)
            << "`concurrency` (= " << concurrency
            << ") must be at least twice as large as `batch_size` (= " << batch_size << ").";
    }

    {
        if (no_duplicate) {
            Seed_ seed = Kanachan::getRandomSeed();
            std::seed_seq ss(seed.cbegin(), seed.cend());
            std::mt19937 urng(ss);

            for (std::size_t i = 0u; i < num_simulation_sets; ++i) {
                std::array<std::pair<std::uint_fast8_t, bool>, 4u> data{
                    std::pair<std::uint_fast8_t, bool>{ baseline_grade, false },
                    std::pair<std::uint_fast8_t, bool>{ baseline_grade, false },
                    std::pair<std::uint_fast8_t, bool>{
                        one_versus_three ? baseline_grade : proposed_grade, !one_versus_three },
                    std::pair<std::uint_fast8_t, bool>{ proposed_grade, true }
                };
                std::shuffle(data.begin(), data.end(), urng);
                Grades_ grades{ data[0u].first, data[1u].first, data[2u].first, data[3u].first };
                SeatArrangement_ seat_arrangement{
                    data[0u].second, data[1u].second, data[2u].second, data[3u].second };
                game_initializers_.emplace_back(grades, seat_arrangement, Kanachan::getRandomSeed());
            }
        }
        else {
            for (std::size_t i = 0u; i < num_simulation_sets; ++i) {
                if (one_versus_three) {
                    Seed_ seed = Kanachan::getRandomSeed();
                    game_initializers_.emplace_back(
                        Grades_{ baseline_grade, baseline_grade, baseline_grade, proposed_grade },
                        SeatArrangement_{ false, false, false, true }, seed);
                    game_initializers_.emplace_back(
                        Grades_{ baseline_grade, baseline_grade, proposed_grade, baseline_grade },
                        SeatArrangement_{ false, false, true, false }, seed);
                    game_initializers_.emplace_back(
                        Grades_{ baseline_grade, proposed_grade, baseline_grade, baseline_grade },
                        SeatArrangement_{ false, true, false, false }, seed);
                    game_initializers_.emplace_back(
                        Grades_{ proposed_grade, baseline_grade, baseline_grade, baseline_grade },
                        SeatArrangement_{ true, false, false, false }, seed);
                }
                else {
                    Seed_ seed = Kanachan::getRandomSeed();
                    game_initializers_.emplace_back(
                        Grades_{ baseline_grade, baseline_grade, proposed_grade, proposed_grade },
                        SeatArrangement_{ false, false, true, true }, seed);
                    game_initializers_.emplace_back(
                        Grades_{ baseline_grade, proposed_grade, baseline_grade, proposed_grade },
                        SeatArrangement_{ false, true, false, true }, seed);
                    game_initializers_.emplace_back(
                        Grades_{ baseline_grade, proposed_grade, proposed_grade, baseline_grade },
                        SeatArrangement_{ false, true, true, false }, seed);
                    game_initializers_.emplace_back(
                        Grades_{ proposed_grade, baseline_grade, baseline_grade, proposed_grade },
                        SeatArrangement_{ true, false, false, true }, seed);
                    game_initializers_.emplace_back(
                        Grades_{ proposed_grade, baseline_grade, proposed_grade, baseline_grade },
                        SeatArrangement_{ true, false, true, false }, seed);
                    game_initializers_.emplace_back(
                        Grades_{ proposed_grade, proposed_grade, baseline_grade, baseline_grade },
                        SeatArrangement_{ true, true, false, false }, seed);
                }
            }
        }
    }

    for (std::size_t i = 0u; i < concurrency; ++i) {
        threads_.emplace_back(&Impl_::threadMain_, this);
    }
}

void Simulator::Impl_::threadMain_(std::stop_token stop_token)
try {
    if (Py_IsInitialized() == 0) {
        KANACHAN_THROW<std::runtime_error>("Python interpreter must be initialized.");
    }

    for (;;) {
        Grades_ grades;
        SeatArrangement_ seats;
        Seed_ seed;
        {
            std::unique_lock lock(mtx_);
            if (game_initializers_.empty()) {
                std::size_t const num_alive_threads = --num_alive_threads_;
                lock.unlock();
                p_decision_maker_->shrinkBatchSizeToFitNumThreads(num_alive_threads);
                return;
            }

            std::tie(grades, seats, seed) = std::move(game_initializers_.back());
            game_initializers_.pop_back();
        }

        std::shared_ptr<Kanachan::GameLog> p_game_log = [&]() {
            std::seed_seq ss(seed.cbegin(), seed.cend());
            std::mt19937 urng(ss);
            Kanachan::Decider baseline_decider = std::bind_front(
                &DecisionMaker::operator(), p_decision_maker_, false);
            Kanachan::Decider proposed_decider = std::bind_front(
                &DecisionMaker::operator(), p_decision_maker_, true);
            std::array<Decider, 4u> deciders{
                seats[0u] ? proposed_decider : baseline_decider,
                seats[1u] ? proposed_decider : baseline_decider,
                seats[2u] ? proposed_decider : baseline_decider,
                seats[3u] ? proposed_decider : baseline_decider
            };
            std::vector<Kanachan::Paishan> dummy_paishan_list;
            std::shared_ptr<Kanachan::GameLog> p_game_log = Kanachan::simulateGame(
                urng, room_, dong_feng_zhan_, deciders, grades, dummy_paishan_list, stop_token);
            p_game_log->setWithProposedModel(seats);
            return p_game_log;
        }();

        {
            std::scoped_lock lock(mtx_);
            game_logs_.push_back(p_game_log);
        }

        {
            Kanachan::GIL::RecursiveLock gil;
            progress_();
        }
    }
}
catch (Kanachan::ThreadTermination const &) {
    // This is graceful termination, so there is nothing to do.
}
catch (python::error_already_set const &) {
    try {
        Kanachan::translatePythonException();
    } catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
    }
    std::abort();
}
catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
    std::abort();
}

python::list Simulator::Impl_::run()
{
    KANACHAN_ASSERT((PyGILState_Check() == 1));

    p_decision_maker_->run();

    {
        Kanachan::GIL::RecursiveRelease gil_release;
        for (auto &thread : threads_) {
            KANACHAN_ASSERT((!thread.joinable()));
        }
    }

    python::list game_logs;
    for (std::shared_ptr<Kanachan::GameLog> p_game_log : game_logs_) {
        game_logs.append(p_game_log);
    }
    return game_logs;
}

} // namespace Kanachan
