#if !defined(KANACHAN_SIMULATION_GAME_LOG_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_GAME_LOG_HPP_INCLUDE_GUARD

#include "simulation/round_result.hpp"
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <vector>
#include <array>
#include <cstdint>


namespace Kanachan{

class GameLog
{
private:
  class Decision_
  {
  public:
    Decision_(
      std::vector<std::uint_fast16_t> &&sparse, std::vector<std::uint_fast16_t> &&numeric,
      std::vector<std::uint_fast16_t> &&progression, std::vector<std::uint_fast16_t> &&candidates,
      std::uint_fast16_t action);

    std::vector<std::uint_fast16_t> const &getSparse() const noexcept;

    std::vector<std::uint_fast16_t> const &getNumeric() const noexcept;

    std::vector<std::uint_fast16_t> const &getProgression() const noexcept;

    std::vector<std::uint_fast16_t> const &getCandidates() const noexcept;

    std::uint_fast8_t getActionIndex() const noexcept;

  private:
    std::vector<std::uint_fast16_t> sparse_;
    std::vector<std::uint_fast16_t> numeric_;
    std::vector<std::uint_fast16_t> progression_;
    std::vector<std::uint_fast16_t> candidates_;
    std::uint_fast8_t action_index_;
  };

  using Decisions_ = std::vector<Decision_>;

  using RoundResults_ = std::vector<Kanachan::RoundResult>;

public:
  GameLog();

  GameLog(GameLog const &) = delete;

  GameLog &operator=(GameLog const &) = delete;

  void onBeginningOfRound();

  void onDecision(
    std::uint_fast8_t seat, std::vector<std::uint_fast16_t> &&sparse,
    std::vector<std::uint_fast16_t> &&numeric, std::vector<std::uint_fast16_t> &&progression,
    std::vector<std::uint_fast16_t> &&candidates, std::uint_fast16_t action);

  void onEndOfRound(std::array<Kanachan::RoundResult, 4u> const &round_results);

  void onEndOfGame(std::array<std::int_fast32_t, 4u> const &scores);

  void setWithProposedModel(std::array<bool, 4u> const &with_proposed_model);

  boost::python::list getResult() const;

  boost::python::dict getEpisode(std::uint_fast8_t seat) const;

private:
  std::array<Decisions_, 4u> decisions_;
  std::array<RoundResults_, 4u> round_results_;
  std::array<std::int_fast32_t, 4u> game_scores_;
  std::array<bool, 4u> with_proposed_model_;
};

}

#endif
