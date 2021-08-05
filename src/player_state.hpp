#if !defined(KANACHAN_PLAYER_STATE_HPP_INCLUDE_GUARD)
#define KANACHAN_PLAYER_STATE_HPP_INCLUDE_GUARD

#include "throw.hpp"

#include "mahjongsoul.pb.h"
#include <iosfwd>
#include <array>
#include <limits>
#include <cstdint>


namespace Kanachan{

class PostZimoAction;

class PlayerState
{
private:
  friend class Kanachan::PostZimoAction;

public:
  PlayerState(
    std::uint_fast8_t seat,
    std::uint_fast32_t mode_id,
    std::uint_fast32_t level,
    std::uint_fast8_t game_rank,
    std::int_fast32_t game_score,
    std::int_fast32_t delta_grading_point);

  PlayerState(PlayerState const &) = default;

  PlayerState &operator=(PlayerState const &) = delete;

  void onNewRound(lq::RecordNewRound const &record);

  void onZimo(lq::RecordDealTile const &record);

private:
  std::uint_fast8_t getFuluTail_() const;

  void handIn_();

public:
  void onDapai(lq::RecordDiscardTile const &record);

private:
  void onMyChi_(std::array<std::uint_fast8_t, 3u> const &tiles);

  void onMyPeng_(std::array<std::uint_fast8_t, 3u> const &tiles);

  void onMyDaminggang_(std::array<std::uint_fast8_t, 4u> const &tiles);

public:
  void onChiPengGang(lq::RecordChiPengGang const &record);

private:
  void onMyJiagang_(std::uint_fast8_t tile);

  void onMyAngang_(std::uint_fast8_t tile);

public:
  void onGang(lq::RecordAnGangAddGang const &record);

  std::int_fast32_t getInitialScore() const;

  std::int_fast32_t getCurrentScore() const;

private:
  static constexpr std::array<uint_fast8_t, 6u> fulu_offsets_{
    0u, 90u, 127u, 164u, 201u, 238u
  };
  static constexpr std::uint_fast8_t num_types_of_fulu_ = fulu_offsets_[5u];

  std::uint_fast8_t room_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 5u)
  std::uint_fast8_t dongfeng_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::uint_fast8_t banzuang_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::uint_fast8_t seat_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 4u)
  std::uint_fast8_t qijia_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 4u)
  std::uint_fast8_t chang_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 3u)
  std::uint_fast8_t ju_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 4u)
  std::uint_fast8_t ben_ = std::numeric_limits<std::uint_fast8_t>::max(); // integral
  std::uint_fast8_t liqibang_ = std::numeric_limits<std::uint_fast8_t>::max(); // integral
  std::array<std::uint_fast8_t, 5u> dora_indicators_{
    std::numeric_limits<std::uint_fast8_t>::max(), // [0, 37u)
    std::numeric_limits<std::uint_fast8_t>::max(), // [0, 37u), optional
    std::numeric_limits<std::uint_fast8_t>::max(), // [0, 37u), optional
    std::numeric_limits<std::uint_fast8_t>::max(), // [0, 37u), optional
    std::numeric_limits<std::uint_fast8_t>::max() // [0, 37u), optional
  };
  std::uint_fast8_t count_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 70u)
  std::uint_fast8_t level_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 16u)
  std::uint_fast8_t rank_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 4u)
  std::int_fast32_t initial_score_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
  std::int_fast32_t score_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
  std::array<std::uint_fast8_t, 136u> hand_{}; // combination
  std::uint_fast8_t zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 37u), optional
  std::uint_fast8_t first_zimo_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::uint_fast8_t lingshang_zimo_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::uint_fast8_t yifa_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::uint_fast8_t chipeng_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 4u)
  std::array<std::uint_fast16_t, 24u> he_; // [0u, 37u * 8u) = [0, 296u), optional
  std::array<std::uint_fast8_t, 4u> fulus_; // [0u, num_types_of_fulus_) = [0u, 238u), optional
  std::uint_fast8_t game_rank_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0, 4u)
  std::int_fast32_t game_score_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
  std::int_fast32_t delta_grading_point_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
}; // class PlayerState

} // namespace Kanachan

#endif // !defined(KANACHAN_PLAYER_STATE_HPP_INCLUDE_GUARD)
