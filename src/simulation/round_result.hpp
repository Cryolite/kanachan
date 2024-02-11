#if !defined(KANACHAN_SIMULATION_ROUND_RESULT_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_ROUND_RESULT_HPP_INCLUDE_GUARD

#include <cstdint>


namespace Kanachan{

class RoundResult
{
public:
  RoundResult();

  // `type ==  0`: 親の自摸和
  // `type ==  1`: 子の自摸和
  // `type ==  2`: 子⇒親の被自摸和
  // `type ==  3`: 親=>子の被自摸和
  // `type ==  4`: 子=>子の被自摸和
  // `type ==  5`: 親の栄和
  // `type ==  6`: 子の栄和
  // `type ==  7`: 子⇒親の放銃
  // `type ==  8`: 親=>子の放銃
  // `type ==  9`: 子=>子の放銃
  // `type == 10`: 横移動
  // `type == 11`: 荒牌平局（不聴）
  // `type == 12`: 荒牌平局（聴牌）
  // `type == 13`: 荒牌平局（流し満貫）
  // `type == 14`: 途中流局
  void setType(std::uint_fast8_t type);

  void setInLizhi(bool in_lizhi);

  void setHasFulu(bool has_fulu);

  void setRoundDeltaScore(std::int_fast32_t round_delta_score);

  void setRoundScore(std::int_fast32_t round_score);

  std::uint_fast8_t getType() const noexcept;

  bool getInLizhi() const noexcept;

  bool getHasFulu() const noexcept;

  std::int_fast32_t getRoundDeltaScore() const noexcept;

  std::int_fast32_t getRoundScore() const noexcept;

private:
  std::uint_fast8_t type_;
  bool in_lizhi_;
  bool has_fulu_;
  std::int_fast32_t round_delta_score_;
  std::int_fast32_t round_score_;
};

}

#endif
