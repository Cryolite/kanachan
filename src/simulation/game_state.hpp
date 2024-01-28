#if !defined(KANACHAN_SIMULATION_GAME_STATE_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_GAME_STATE_HPP_INCLUDE_GUARD

#include "simulation/decision_maker.hpp"
#include "simulation/game_log.hpp"
#include <stop_token>
#include <array>
#include <utility>
#include <memory>
#include <cstdint>


namespace Kanachan{

class GameState
{
public:
  using Seat = std::pair<std::uint_fast8_t, std::shared_ptr<Kanachan::DecisionMaker>>;

public:
  GameState(
    std::uint_fast8_t room, bool dong_feng_zhan, Kanachan::Deciders deciders,
    std::array<std::uint_fast8_t, 4u> const &grades, std::stop_token stop_token);

  GameState(GameState const &) = delete;

  GameState(GameState &&) = delete;

  GameState &operator=(GameState const &) = delete;

  GameState &operator=(GameState &&) = delete;

public:
  std::uint_fast8_t getRoom() const;

  bool isDongfengZhan() const;

  std::uint_fast8_t getPlayerGrade(std::uint_fast8_t seat) const;

  std::uint_fast8_t getChang() const;

  std::uint_fast8_t getJu() const;

  std::uint_fast8_t getBenChang() const;

  std::uint_fast8_t getNumLizhiDeposits() const;

  std::int_fast32_t getPlayerScore(std::uint_fast8_t seat) const;

  std::uint_fast8_t getPlayerRanking(std::uint_fast8_t seat) const;

public:
  std::uint_fast16_t selectAction(
    std::uint_fast8_t seat, std::vector<std::uint_fast16_t> &&sparse,
    std::vector<std::uint_fast32_t> &&numeric, std::vector<std::uint_fast16_t> &&progression,
    std::vector<std::uint_fast16_t> &&candidates, Kanachan::GameLog &game_log) const;

public:
  void onSuccessfulLizhi(std::uint_fast8_t seat);

  void addPlayerScore(std::uint_fast8_t seat, std::int_fast32_t score);

  enum struct RoundEndStatus : std::uint_fast8_t
  {
    hule = 0u,
    huangpai_pingju = 1u,
    liuju = 2u
  }; // enum struct RoundEndStatus

  void onLianzhuang(RoundEndStatus round_end_status);

  void onLunzhuang(RoundEndStatus round_end_status);

private:
  std::uint_fast8_t room_;
  bool dong_feng_zhan_;
  Kanachan::Deciders deciders_;
  std::array<std::uint_fast8_t, 4u> grades_;
  std::uint_fast8_t chang_ = 0u;
  std::uint_fast8_t ju_ = 0u;
  std::uint_fast8_t ben_chang_ = 0u;
  std::uint_fast8_t lizhi_deposits_ = 0u;
  std::array<std::int_fast32_t, 4u> scores_{ 25000, 25000, 25000, 25000 };
  std::stop_token stop_token_;
}; // class GameState

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_GAME_STATE_HPP_INCLUDE_GUARD)
