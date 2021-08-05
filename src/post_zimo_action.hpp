#if !defined(KANACHAN_POST_ZIMO_ACTION_HPP_INCLUDE_GUARD)
#define KANACHAN_POST_ZIMO_ACTION_HPP_INCLUDE_GUARD

#include "player_state.hpp"
#include "utility.hpp"
#include <iosfwd>
#include <array>
#include <limits>
#include <cstdint>


namespace Kanachan{

namespace Detail_{

class Serializer;

} // namespace Detail_

class PostZimoAction
{
private:
  PostZimoAction(
    std::array<Kanachan::PlayerState, 4u> const &states,
    std::uint_fast8_t seat);

public:
  PostZimoAction(
    std::array<Kanachan::PlayerState, 4u> const &states,
    lq::RecordDiscardTile const &record);

  PostZimoAction(
    std::array<Kanachan::PlayerState, 4u> const &states,
    lq::RecordAnGangAddGang const &record);

  PostZimoAction(
    std::array<Kanachan::PlayerState, 4u> const &states,
    lq::HuleInfo const &record);

  PostZimoAction(
    std::array<Kanachan::PlayerState, 4u> const &states,
    lq::RecordLiuJu const &record);

  PostZimoAction &operator=(PostZimoAction const &rhs) = default;

  void setDeltaRoundScore(std::int_fast32_t delta_round_score);

  void encode(std::string const &uuid, std::ostream &os) const;

private:
  class Tajia_
  {
  public:
    Tajia_();

    void initialize(
      std::uint_fast8_t level,
      std::uint_fast8_t rank,
      std::int_fast32_t score,
      std::uint_fast8_t yifa,
      std::array<std::uint_fast16_t, 24u> const &he,
      std::array<std::uint_fast8_t, 4u> const &fulus);

    void encode(Detail_::Serializer &s) const;

  private:
    std::uint_fast8_t level_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 16u)
    std::uint_fast8_t rank_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 4u)
    std::int_fast32_t score_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
    std::uint_fast8_t yifa_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
    std::array<std::uint_fast16_t, 24u> he_; // [0u, 37u * 8u) = [0, 296u), optional
    std::array<std::uint_fast8_t, 4u> fulus_; // [0u, num_types_of_fulus_) = [0u, 238u), optional
  }; // class Tajia_

private:
  std::uint_fast8_t room_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 5u)
  std::uint_fast8_t dongfeng_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::uint_fast8_t banzuang_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
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
  std::int_fast32_t score_ = std::numeric_limits<std::int_fast32_t>::max(); // integral
  std::array<std::uint_fast8_t, 136u> hand_{}; // combination
  std::uint_fast8_t zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 37u), optional
  std::uint_fast8_t first_zimo_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::uint_fast8_t lingshang_zimo_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::uint_fast8_t yifa_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u)
  std::array<std::uint_fast16_t, 24u> he_; // [0u, 37u * 8u) = [0, 296u), optional
  std::array<std::uint_fast8_t, 4u> fulus_; // [0u, num_types_of_fulus_) = [0u, 238u), optional
  std::array<Tajia_, 3u> tajias_;
  std::uint_fast8_t dapai_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 37u * 4u), optional
  std::uint_fast8_t gang_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 74u), optional
  std::uint_fast8_t hule_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u), optional
  std::uint_fast8_t liuju_ = std::numeric_limits<std::uint_fast8_t>::max(); // [0u, 1u), optional
  std::int_fast32_t delta_round_score_ = std::numeric_limits<std::int_fast32_t>::max();
  std::uint_fast8_t game_rank_ = std::numeric_limits<std::uint_fast8_t>::max();
  std::int_fast32_t game_score_ = std::numeric_limits<std::int_fast32_t>::max();
  std::int_fast32_t delta_grading_point_ = std::numeric_limits<std::int_fast32_t>::max();
}; // class PostZimoAction

} // namespace Kanachan


#endif // !defined(KANACHAN_POST_ZIMO_ACTION_HPP_INCLUDE_GUARD)
