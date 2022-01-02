#if !defined(KANACHAN_ANNOTATION_PLAYER_STATE_HPP_INCLUDE_GUARD)
#define KANACHAN_ANNOTATION_PLAYER_STATE_HPP_INCLUDE_GUARD

#include "common/throw.hpp"
#include "common/mahjongsoul.pb.h"
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

  void onLiuju(
    lq::RecordLiuJu const &record, std::uint_fast8_t seat,
    lq::RecordNewRound const &next_record);

  std::uint_fast8_t getSeat() const;

  std::uint_fast8_t getLeftTileCount() const;

  std::uint_fast8_t getLevel() const;

  std::uint_fast8_t getRank(
    std::array<PlayerState, 4u> const &player_states) const;

  std::int_fast32_t getInitialScore() const;

  std::int_fast32_t getCurrentScore() const;

  std::vector<std::uint_fast8_t> getDiscardableTiles() const;

  std::uint_fast8_t getZimopai() const;

  std::uint_fast8_t getGameRank() const;

  std::int_fast32_t getGameScore() const;

  std::int_fast32_t getDeltaGradingPoint() const;

  void print(std::array<PlayerState, 4u> const &player_states,
             std::ostream &os) const;

private:
  static constexpr std::array<std::uint_fast8_t, 38u> hand_offset_{
      0u,   1u,   5u ,  9u,  13u,  17u,  20u,  24u,  28u,  32u,
     36u,  37u,  41u,  45u,  49u,  53u,  56u,  60u,  64u,  68u,
     72u,  73u,  77u,  81u,  85u,  89u,  92u,  96u, 100u, 104u,
    108u, 112u, 116u, 120u, 124u, 128u, 132u, 136u
  };

  // Match state
  std::uint_fast8_t room_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 5u)
  std::uint_fast8_t num_rounds_type_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 2u)
  std::uint_fast8_t seat_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 4u)

  // Round state
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

  // Player state
  std::uint_fast8_t level_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 16u)
  std::int_fast32_t initial_score_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
  std::int_fast32_t score_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
  std::array<std::uint_fast8_t, 136u> hand_{}; // combination
  std::uint_fast8_t zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 37u), optional

  // Match result
  std::uint_fast8_t game_rank_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0, 4u)
  std::int_fast32_t game_score_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
  std::int_fast32_t delta_grading_point_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
}; // class PlayerState

} // namespace Kanachan

#endif // !defined(KANACHAN_ANNOTATION_PLAYER_STATE_HPP_INCLUDE_GUARD)
