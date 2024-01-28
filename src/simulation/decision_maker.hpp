#if !defined(KANACHAN_SIMULATION_DECISION_MAKER_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_DECISION_MAKER_HPP_INCLUDE_GUARD

#include "simulation/game_log.hpp"
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <stop_token>
#include <string>
#include <array>
#include <memory>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

class DecisionMaker
{
private:
    class Impl_;

public:
    DecisionMaker(
        boost::python::object device, boost::python::object dtype,
        boost::python::object baseline_model, boost::python::list baseline_keys_to_be_deleted,
        boost::python::object proposed_model, boost::python::list proposed_keys_to_be_deleted,
        std::size_t batch_size);

    DecisionMaker(DecisionMaker const &) = delete;

    DecisionMaker &operator=(DecisionMaker const &) = delete;

    void shrinkBatchSizeToFitNumThreads(std::size_t num_threads);

    std::uint_fast16_t operator()(
        bool proposed, std::uint_fast8_t seat, std::vector<std::uint_fast16_t> &&sparse,
        std::vector<std::uint_fast32_t> &&numeric, std::vector<std::uint_fast16_t> &&progression,
        std::vector<std::uint_fast16_t> &&candidates, std::stop_token stop_token,
        Kanachan::GameLog &game_log);

    void run();

private:
    std::shared_ptr<Impl_> p_impl_;
}; // class DecisionMaker

using Decider = std::function<
    std::uint_fast16_t(
        std::uint_fast8_t, std::vector<std::uint_fast16_t> &&,
        std::vector<std::uint_fast32_t> &&, std::vector<std::uint_fast16_t> &&,
        std::vector<std::uint_fast16_t> &&, std::stop_token, Kanachan::GameLog &)>;
using Deciders = std::array<Decider, 4u>;

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_DECISION_MAKER_HPP_INCLUDE_GUARD)
