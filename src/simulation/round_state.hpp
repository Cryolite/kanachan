#if !defined(KANACHAN_SIMULATION_ROUND_STATE_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_ROUND_STATE_HPP_INCLUDE_GUARD

#include "simulation/game_log.hpp"
#include "simulation/shoupai.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include <vector>
#include <array>
#include <utility>
#include <limits>
#include <cstdint>


namespace Kanachan{

class RoundState
{
public:
  RoundState(
    std::vector<std::uint_least32_t> const &seed, Kanachan::GameState &game_state,
    Kanachan::Paishan const *p_test_paishan);

  RoundState(RoundState const &) = delete;

  RoundState(RoundState &&) = delete;

  RoundState &operator=(RoundState const &) = delete;

  RoundState &operator=(RoundState &&) = delete;

public:
  std::uint_fast8_t getPlayerGrade(std::uint_fast8_t seat) const;

  std::uint_fast8_t getChang() const;

  std::uint_fast8_t getJu() const;

  std::uint_fast8_t getBenChang() const;

  std::uint_fast8_t getNumLizhiDeposits() const;

  std::int_fast32_t getPlayerScore(std::uint_fast8_t seat) const;

  std::uint_fast8_t getPlayerRanking(std::uint_fast8_t seat) const;

  std::uint_fast8_t getDoraIndicator(std::uint_fast8_t index) const;

  std::uint_fast8_t getNumLeftTiles() const;

  std::int_fast32_t getPlayerDeltaScore(std::uint_fast8_t seat) const;

  bool isPlayerMenqian(std::uint_fast8_t seat) const;

  bool isPlayerTingpai(std::uint_fast8_t seat) const;

  bool checkPlayerLiujuManguan(std::uint_fast8_t seat) const;

  bool checkSifengLianda() const;

  bool checkSigangSanle() const;

  bool checkSijiaLizhi() const;

private:
  std::pair<std::uint_fast8_t, std::uint_fast8_t> getLastDapai_() const;

  std::uint_fast8_t drawLingshangPai_();

  std::pair<std::vector<std::uint_fast16_t>, std::vector<std::uint_fast32_t>>
  constructFeatures_(std::uint_fast8_t seat, std::uint_fast8_t zimo_tile) const;

  std::uint_fast16_t selectAction_(
    std::uint_fast8_t seat, std::vector<std::uint_fast16_t> &&sparse,
    std::vector<std::uint_fast32_t> &&numeric, std::vector<uint_fast16_t> &&progression,
    std::vector<std::uint_fast16_t> &&candidates, Kanachan::GameLog &game_log) const;

  long encodeToolConfig_(std::uint_fast8_t seat, bool rong) const;

  std::vector<std::uint_fast8_t> constructDoraIndicators_(std::uint_fast8_t seat) const;

  std::pair<std::uint_fast8_t, std::uint_fast8_t> checkDaSanyuanPao_() const;

  std::pair<std::uint_fast8_t, std::uint_fast8_t> checkDaSixiPao_() const;

  std::pair<std::uint_fast8_t, std::uint_fast8_t> calculateHand_(
    std::uint_fast8_t seat, std::uint_fast8_t hupai, long config) const;

  void settleLizhiDeposits_();

public:
  std::pair<std::uint_fast16_t, std::uint_fast8_t> onZimo(Kanachan::GameLog &game_log);

  std::pair<std::uint_fast8_t, std::uint_fast16_t>
  onDapai(std::uint_fast8_t tile, bool moqi, bool lizhi, Kanachan::GameLog &game_log);

  std::uint_fast16_t onChi(std::uint_fast8_t encode, Kanachan::GameLog &game_log);

  std::uint_fast16_t onPeng(std::uint_fast8_t encode, Kanachan::GameLog &game_log);

  void onDaminggang(Kanachan::GameLog &game_log);

  std::uint_fast16_t onAngang(
    std::uint_fast8_t zimo_tile, std::uint_fast8_t encode, Kanachan::GameLog &game_log);

  std::uint_fast16_t onJiagang(
    std::uint_fast8_t zimo_tile, std::uint_fast8_t encode, Kanachan::GameLog &game_log);

  bool onHule(std::uint_fast8_t zimo_tile, Kanachan::GameLog &game_log);

  bool onHuangpaiPingju(Kanachan::GameLog &game_log);

  void onLiuju(Kanachan::GameLog &game_log);

private:
  Kanachan::GameState &game_state_;
  std::array<std::int_fast32_t, 4u> initial_scores_{
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max()
  };
  Kanachan::Paishan paishan_;
  std::uint_fast8_t zimo_index_ = 4u * 13u;
  std::uint_fast8_t lingshang_zimo_count_ = 0u;
  std::uint_fast8_t gang_dora_count_ = 0u;
  std::array<Kanachan::Shoupai, 4u> shoupai_list_;
  std::uint_fast8_t seat_;
  std::array<std::uint_fast8_t, 4u> lizhi_list_{ 0u, 0u, 0u, 0u };
  std::uint_fast8_t lizhi_delayed_ = 0u;
  bool angang_dora_delayed_ = false;
  bool minggang_dora_delayed_ = false;
  std::array<bool, 4u> first_zimo_ = { true, true, true, true };
  std::array<bool, 4u> yifa_ = { false, false, false, false };
  bool lingshang_kaihua_delayed_ = false;
  bool qianggang_delayed_ = false;
  std::array<bool, 4u> rong_delayed_ = { false, false, false, false };
  std::vector<std::uint_fast16_t> progression_;
}; // class RoundState

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_ROUND_STATE_HPP_INCLUDE_GUARD)
