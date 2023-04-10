#if !defined(KANACHAN_SIMULATION_DECISION_MAKER_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_DECISION_MAKER_HPP_INCLUDE_GUARD

#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <stop_token>
#include <string>
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
        std::string const &device, boost::python::object dtype, boost::python::object model,
        std::size_t batch_size);

    DecisionMaker(DecisionMaker const &) = delete;

    DecisionMaker &operator=(DecisionMaker const &) = delete;

    void shrinkBatchSizeToFitNumThreads(std::size_t num_threads);

    std::uint_fast16_t operator()(
        std::vector<std::uint_fast16_t> &&sparse, std::vector<std::uint_fast32_t> &&numeric,
        std::vector<std::uint_fast16_t> &&progression, std::vector<std::uint_fast16_t> &&candidates,
        std::stop_token stop_token);

    void join();

private:
    std::shared_ptr<Impl_> p_impl_;
}; // class DecisionMaker

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_DECISION_MAKER_HPP_INCLUDE_GUARD)
