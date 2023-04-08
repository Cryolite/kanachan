#include "simulation/simulator.hpp"

#include "simulation/game.hpp"
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
    using Seat_ = std::pair<std::uint_fast8_t, std::shared_ptr<Kanachan::DecisionMaker>>;
    using Seats_ = std::array<Seat_, 4u>;

public:
    Impl_(
        std::string const &device, python::object dtype,
        std::uint_fast8_t baseline_grade, python::object baseline_model,
        std::uint_fast8_t proposed_grade, python::object proposed_model,
        unsigned long simulation_mode, std::size_t num_simulation_sets,
        std::size_t batch_size, std::size_t concurrency);

    Impl_(Impl_ const &) = delete;

    Impl_ &operator=(Impl_ const &) = delete;

private:
    void threadMain_(std::stop_token stop_token);

public:
    python::list run();

private:
    bool dong_feng_zhan_;
    std::uint_fast8_t room_;
    std::shared_ptr<Kanachan::DecisionMaker> p_baseline_decision_maker_;
    std::shared_ptr<Kanachan::DecisionMaker> p_proposed_decision_maker_;
    std::vector<std::jthread> threads_;
    std::vector<std::vector<std::uint_least32_t>> seeds_;
    std::vector<Seats_> seats_list_;
    std::vector<python::dict> results_;
    std::size_t num_alive_threads_;
    std::mutex mtx_;
}; // class Simulator::Impl_

Simulator::Simulator(
    std::string const &device, python::object dtype,
    std::uint_fast8_t baseline_grade, python::object baseline_model,
    std::uint_fast8_t proposed_grade, python::object proposed_model,
    unsigned long simulation_mode, std::size_t num_simulation_sets,
    std::size_t batch_size, std::size_t concurrency)
    : p_impl_(std::make_shared<Impl_>(
        device, dtype, baseline_grade, baseline_model, proposed_grade, proposed_model,
        simulation_mode, num_simulation_sets, batch_size, concurrency))
{}

python::list Simulator::run()
{
    KANACHAN_ASSERT((!!p_impl_));
    return p_impl_->run();
}

Simulator::Impl_::Impl_(
    std::string const &device, python::object dtype,
    std::uint_fast8_t baseline_grade, python::object baseline_model,
    std::uint_fast8_t proposed_grade, python::object proposed_model,
    unsigned long simulation_mode, std::size_t num_simulation_sets,
    std::size_t batch_size, std::size_t concurrency)
    : dong_feng_zhan_(simulation_mode & 2u)
    , room_(std::numeric_limits<std::uint_fast8_t>::max())
    , p_baseline_decision_maker_(
          std::make_shared<Kanachan::DecisionMaker>(device, dtype, baseline_model, batch_size))
    , p_proposed_decision_maker_(
          std::make_shared<Kanachan::DecisionMaker>(device, dtype, proposed_model, batch_size))
    , threads_()
    , seeds_()
    , seats_list_()
    , results_()
    , num_alive_threads_(concurrency)
    , mtx_()
{
    if (num_simulation_sets == 0u) {
        KANACHAN_THROW<std::invalid_argument>("`num_simulation_sets` must be a positive integer.");
    }

    bool const no_duplicate = (simulation_mode & 1u);
    bool const one_versus_three = (simulation_mode & 4u);

    if ((simulation_mode & 8u) != 0u) {
        room_ = 0u;
    }
    if ((simulation_mode & 16u) != 0u) {
        if (room_ != std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::invalid_argument>("simulation_mode: Multiple rooms specified.");
        }
        room_ = 1u;
    }
    if ((simulation_mode & 32u) != 0u) {
        if (room_ != std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::invalid_argument>("simulation_mode: Multiple rooms specified.");
        }
        room_ = 2u;
    }
    if ((simulation_mode & 64u) != 0u) {
        if (room_ != std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::invalid_argument>("simulation_mode: Multiple rooms specified.");
        }
        room_ = 3u;
    }
    if ((simulation_mode & 128u) != 0u) {
        if (room_ != std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::invalid_argument>("simulation_mode: Multiple rooms specified.");
        }
        room_ = 4u;
    }
    if (room_ == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("simulation_mode: Room not specified.");
    }

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
        using Seed = std::vector<std::uint_least32_t>;
        using Flags = std::array<bool, 4u>;

        auto const append = [&](Seed const &seed, Flags const flags) {
            seeds_.push_back(seed);

            Seat_ const baseline_seat(baseline_grade, p_baseline_decision_maker_);
            Seat_ const proposed_seat(proposed_grade, p_proposed_decision_maker_);
            Seats_ seats = {
                flags[0u] ? proposed_seat : baseline_seat,
                flags[1u] ? proposed_seat : baseline_seat,
                flags[2u] ? proposed_seat : baseline_seat,
                flags[3u] ? proposed_seat : baseline_seat
            };
            seats_list_.push_back(seats);
        };

        if (no_duplicate) {
            std::vector<std::uint_least32_t> seed = Kanachan::getRandomSeed();
            std::seed_seq ss(seed.cbegin(), seed.cend());
            std::mt19937 urng(ss);

            for (std::size_t i = 0u; i < num_simulation_sets; ++i) {
                std::array<bool, 4u> flags = { false, false, !one_versus_three, true };
                std::shuffle(flags.begin(), flags.end(), urng);
                append(Kanachan::getRandomSeed(), flags);
            }
        }
        else {
            for (std::size_t i = 0u; i < num_simulation_sets; ++i) {
                if (one_versus_three) {
                    Seed seed = Kanachan::getRandomSeed();
                    append(seed, { false, false, false, true });
                    append(seed, { false, false, true, false });
                    append(seed, { false, true, false, false });
                    append(seed, { true, false, false, false });
                }
                else {
                    Seed seed = Kanachan::getRandomSeed();
                    append(seed, { false, false, true, true });
                    append(seed, { false, true, false, true });
                    append(seed, { false, true, true, false });
                    append(seed, { true, false, false, true });
                    append(seed, { true, false, true, false });
                    append(seed, { true, true, false, false });
                }
            }
        }
    }

    for (std::size_t i = 0u; i < concurrency; ++i) {
        threads_.emplace_back(&Impl_::threadMain_, this);
    }

    {
        Kanachan::GIL::RecursiveRelease gil_release;
        for (std::jthread &thread : threads_) {
            thread.join();
        }
        p_baseline_decision_maker_->join();
        p_proposed_decision_maker_->join();
    }
    p_baseline_decision_maker_.reset();
    p_proposed_decision_maker_.reset();
}

void Simulator::Impl_::threadMain_(std::stop_token stop_token)
try {
    for (;;) {
        std::vector<std::uint_least32_t> seed;
        Seats_ seats;
        {
            std::scoped_lock lock(mtx_);
            if (seeds_.empty()) {
                --num_alive_threads_;
                if (num_alive_threads_ == 0u) {
                    return;
                }
                if (num_alive_threads_ < p_baseline_decision_maker_->getBatchSize() * 2u - 1u) {
                    std::size_t const log2_new_batch_size = std::log2(num_alive_threads_ + 1u) - 1u;
                    p_baseline_decision_maker_->shrinkBatchSize(1u << log2_new_batch_size);
                }
                if (num_alive_threads_ < p_proposed_decision_maker_->getBatchSize() * 2u - 1u) {
                    std::size_t const log2_new_batch_size = std::log2(num_alive_threads_ + 1u) - 1u;
                    p_proposed_decision_maker_->shrinkBatchSize(1u << log2_new_batch_size);
                }
                return;
            }

            KANACHAN_ASSERT((!seats_list_.empty()));
            seed = std::move(seeds_.back());
            seeds_.pop_back();
            seats = std::move(seats_list_.back());
            seats_list_.pop_back();
        }

        python::dict result = [&]() {
            std::vector<Kanachan::Paishan> dummy_paishan_list;
            python::dict result = Kanachan::simulateGame(
                seed, room_, dong_feng_zhan_, seats, dummy_paishan_list, stop_token);
            {
                Kanachan::GIL::RecursiveLock gil_lock;
                result["proposed"] = python::list();
                result["proposed"].attr("append")(
                    seats[0u].second.get() == p_proposed_decision_maker_.get() ? 1 : 0);
                result["proposed"].attr("append")(
                    seats[1u].second.get() == p_proposed_decision_maker_.get() ? 1 : 0);
                result["proposed"].attr("append")(
                    seats[2u].second.get() == p_proposed_decision_maker_.get() ? 1 : 0);
                result["proposed"].attr("append")(
                    seats[3u].second.get() == p_proposed_decision_maker_.get() ? 1 : 0);
            }
            return result;
        }();

        {
            std::scoped_lock lock(mtx_);
            results_.push_back(result);
            std::cout << results_.size() << '/'
                      << seeds_.size() + num_alive_threads_ + results_.size() - 1u << std::endl;
        }
    }
}
catch (Kanachan::ThreadTermination const &) {
    // This is graceful termination, so there is nothing to do.
}
catch (python::error_already_set const &) {
    for (std::jthread &thread : threads_) {
        thread.request_stop();
    }
    try {
        Kanachan::translatePythonException();
    } catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
        throw;
    }
}
catch (std::exception const &e) {
    for (std::jthread &thread : threads_) {
        thread.request_stop();
    }
    std::cerr << e.what() << std::endl;
    throw;
}

python::list Simulator::Impl_::run()
{
    {
        Kanachan::GIL::RecursiveRelease gil_release;
        for (auto &thread : threads_) {
            KANACHAN_ASSERT((!thread.joinable()));
        }
    }

    python::list results;
    for (python::dict result : results_) {
        results.append(result);
    }
    return results;
}

} // namespace Kanachan
