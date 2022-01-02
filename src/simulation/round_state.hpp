#if !defined(KANACHAN_SIMULATION_ROUND_STATE_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_ROUND_STATE_HPP_INCLUDE_GUARD

#include "simulation/shoupai.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <random>
#include <array>
#include <utility>
#include <limits>
#include <cstdint>


namespace Kanachan{

class RoundState
{
public:
  RoundState(
    std::mt19937 &urng, Kanachan::GameState &game_state,
    Kanachan::Paishan const *p_test_paishan,
    boost::python::object external_tool);

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

  boost::python::list constructFeatures_(
    std::uint_fast8_t seat, std::uint_fast8_t zimo_tile) const;

  std::uint_fast16_t selectAction_(
    std::uint_fast8_t seat, boost::python::list features) const;

  long encodeToolConfig_(std::uint_fast8_t seat, bool rong) const;

  boost::python::list constructDoraIndicators_(std::uint_fast8_t seat) const;

  std::pair<std::uint_fast8_t, std::uint_fast8_t> checkDaSanyuanPao() const;

  std::pair<std::uint_fast8_t, std::uint_fast8_t> calculateHand_(
    std::uint_fast8_t seat, std::uint_fast8_t hupai, long config) const;

  void settleLizhiDeposits_();

public:
  std::pair<std::uint_fast16_t, std::uint_fast8_t> onZimo();

  std::pair<std::uint_fast8_t, std::uint_fast16_t>
  onDapai(std::uint_fast8_t tile, bool moqi, bool lizhi);

  std::uint_fast16_t onChi(std::uint_fast8_t encode);

  std::uint_fast16_t onPeng(std::uint_fast8_t encode);

  void onDaminggang();

  std::uint_fast16_t onAngang(
    std::uint_fast8_t zimo_tile, std::uint_fast8_t encode);

  std::uint_fast16_t onJiagang(
    std::uint_fast8_t zimo_tile, std::uint_fast8_t encode);

  bool onHule(std::uint_fast8_t zimo_tile, boost::python::dict result);

  bool onHuangpaiPingju(boost::python::dict result);

  void onLiuju(boost::python::dict result);

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
  boost::python::list progression_;
}; // class RoundState

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_ROUND_STATE_HPP_INCLUDE_GUARD)
