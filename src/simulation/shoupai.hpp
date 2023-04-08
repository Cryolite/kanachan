#if !defined(KANACHAN_SIMULATION_SHOUPAI_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_SHOUPAI_HPP_INCLUDE_GUARD

#include "simulation/paishan.hpp"
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <vector>
#include <array>
#include <utility>
#include <limits>
#include <cstdint>


namespace Kanachan{

class Shoupai;

void swap(Shoupai &lhs, Shoupai &rhs) noexcept;

class Shoupai
{
public:
  Shoupai(std::uint_fast8_t index, Kanachan::Paishan const &paishan);

  Shoupai(Shoupai const &rhs) = default;

  Shoupai(Shoupai &&rhs) = default;

  ~Shoupai();

  void swap(Shoupai &rhs) noexcept;

  Shoupai &operator=(Shoupai const &rhs);

  Shoupai &operator=(Shoupai &&rhs) noexcept;

public:
  bool isMenqian() const;

  bool isTingpai() const;

  std::uint_fast8_t getNumGangzi() const;

private:
  std::vector<std::uint_fast8_t> getShoupai34_() const;

  std::uint_fast8_t getNumFulu_() const;

  boost::python::list getShoupai136_(boost::python::list fulu_list, std::uint_fast8_t hupai) const;

  boost::python::list getFuluList_() const;

  void updateHupaiList_();

public:
  void appendToFeatures(std::vector<std::uint_fast16_t> &sparse_features) const;

  std::vector<std::uint_fast16_t> getCandidatesOnZimo(
    std::uint_fast8_t zimo_tile, bool first_zimo, bool lizhi_prohibited,
    bool gang_prohibited, long tool_config) const;

  std::vector<std::uint_fast16_t> getCandidatesOnDapai(
    std::uint_fast8_t relseat, std::uint_fast8_t dapai, bool gang_prohibited,
    long tool_config) const;

  std::vector<std::uint_fast16_t> getCandidatesOnChiPeng() const;

  std::vector<std::uint_fast16_t> getCandidatesOnAngang(
    std::uint_fast8_t relseat, std::uint_fast8_t encode) const;

  std::vector<std::uint_fast16_t> getCandidatesOnJiagang(
    std::uint_fast8_t relseat, std::uint_fast8_t encode, long tool_config) const;

  std::pair<std::uint_fast8_t, std::uint_fast8_t> calculateHand(
    std::uint_fast8_t hupai, std::vector<std::uint_fast8_t> const &dora_indicators,
    long tool_config) const;

  void onPostZimo(std::uint_fast8_t zimo_tile, std::uint_fast8_t dapai, bool in_lizhi);

  void onChi(std::uint_fast8_t encode);

  void onPeng(std::uint_fast8_t relseat, std::uint_fast8_t encode);

  void onPostChiPeng(std::uint_fast8_t dapai);

  void onDaminggang(std::uint_fast8_t relseat, std::uint_fast8_t dapai);

  void onAngang(std::uint_fast8_t zimo_tile, std::uint_fast8_t encode);

  void onJiagang(std::uint_fast8_t zimo_tile, std::uint_fast8_t encode);

  void onPostGang(bool in_lizhi);

private:
  std::array<std::uint_fast8_t, 37u> shoupai_ = {
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u, 0u
  };

  // チー: [0, 90)
  // ポン: [90, 210) = 90 + relseat * 40 + peng
  // 大明槓: [210, 321) = 210 + relseat * 37 + tile
  // 暗槓: [321, 355) = 321 + tile'
  // 加槓: [355, 392) = 355 + tile
  std::array<std::uint_fast16_t, 4u> fulu_list_ = {
    std::numeric_limits<std::uint_fast16_t>::max(),
    std::numeric_limits<std::uint_fast16_t>::max(),
    std::numeric_limits<std::uint_fast16_t>::max(),
    std::numeric_limits<std::uint_fast16_t>::max()
  };

  std::uint_fast16_t kuikae_delayed_ = std::numeric_limits<std::uint_fast16_t>::max();

  boost::python::object external_tool_;

  std::vector<std::uint_fast8_t> he_;
  mutable bool tingpai_cache_ = false;
  mutable std::uint_fast8_t xiangting_lower_bound_ = std::numeric_limits<std::uint_fast8_t>::max();
  std::vector<std::uint_fast8_t> hupai_list_{};
  mutable bool zhenting_ = false;
}; // class Shoupai

} // namespace Kanachan

#endif // !defined(KANACHAN_SIMULATION_SHOUPAI_HPP_INCLUDE_GUARD)
