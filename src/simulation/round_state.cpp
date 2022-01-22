#include "simulation/round_state.hpp"

#include "simulation/shoupai.hpp"
#include "simulation/paishan.hpp"
#include "simulation/game_state.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <random>
#include <algorithm>
#include <array>
#include <functional>
#include <utility>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

using std::placeholders::_1;
namespace python = boost::python;

RoundState::RoundState(
  std::mt19937 &urng, Kanachan::GameState &game_state,
  Kanachan::Paishan const *p_test_paishan, python::object external_tool)
  : game_state_(game_state),
    paishan_(p_test_paishan == nullptr ? Kanachan::Paishan(urng) : *p_test_paishan),
    shoupai_list_({
        Kanachan::Shoupai((4u - getJu()) % 4u, paishan_, external_tool),
        Kanachan::Shoupai((5u - getJu()) % 4u, paishan_, external_tool),
        Kanachan::Shoupai((6u - getJu()) % 4u, paishan_, external_tool),
        Kanachan::Shoupai((7u - getJu()) % 4u, paishan_, external_tool)
      }),
    seat_(getJu()),
    progression_()
{
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    initial_scores_[i] = game_state_.getPlayerScore(i);
  }
  progression_.append(0);
}

std::uint_fast8_t RoundState::getPlayerGrade(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  return game_state_.getPlayerGrade(seat);
}

std::uint_fast8_t RoundState::getChang() const
{
  return game_state_.getChang();
}

std::uint_fast8_t RoundState::getJu() const
{
  return game_state_.getJu();
}

std::uint_fast8_t RoundState::getBenChang() const
{
  return game_state_.getBenChang();
}

std::uint_fast8_t RoundState::getNumLizhiDeposits() const
{
  return game_state_.getNumLizhiDeposits();
}

std::int_fast32_t RoundState::getPlayerScore(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  return game_state_.getPlayerScore(seat);
}

std::uint_fast8_t
RoundState::getPlayerRanking(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  return game_state_.getPlayerRanking(seat);
}

std::uint_fast8_t
RoundState::getDoraIndicator(std::uint_fast8_t const index) const
{
  KANACHAN_ASSERT((index < 5u));

  if (index > gang_dora_count_) {
    return std::numeric_limits<std::uint_fast8_t>::max();
  }
  // 牌山の王牌および嶺上牌は各幢の上下が逆になっている．
  return paishan_[131u - 2u * index];
}

std::uint_fast8_t RoundState::getNumLeftTiles() const
{
  KANACHAN_ASSERT((zimo_index_ >= 52u));
  KANACHAN_ASSERT((122u >= zimo_index_ + lingshang_zimo_count_));
  return 122u - zimo_index_ - lingshang_zimo_count_;
}

std::int_fast32_t RoundState::getPlayerDeltaScore(std::uint_fast8_t const seat) const
{
  return game_state_.getPlayerScore(seat) - initial_scores_[seat];
}

bool RoundState::isPlayerMenqian(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));

  Kanachan::Shoupai const &shoupai = shoupai_list_[seat];
  return shoupai.isMenqian();
}

bool RoundState::isPlayerTingpai(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));

  Kanachan::Shoupai const &shoupai = shoupai_list_[seat];
  return shoupai.isTingpai();
}

bool RoundState::checkPlayerLiujuManguan(std::uint_fast8_t const seat) const
{
  for (std::size_t i = 0u; i < python::len(progression_); ++i) {
    python::object o = progression_[i];
    python::extract<long> e(o);
    if (!e.check()) {
      KANACHAN_THROW<std::logic_error>("A type error in `progression_`.");
    }
    long const encode = e();

    if (encode == 0) {
      continue;
    }

    if (5 <= encode && encode <= 596) {
      std::uint_fast8_t const seat_ = (encode - 5) / 148;
      if (seat_ != seat) {
        continue;
      }
      std::uint_fast8_t const tile = (encode - 5 - seat_ * 148) / 4;
      if (tile == 0 || 2 <= tile && tile <= 8) {
        return false;
      }
      if (tile == 10 || 12 <= tile && tile <= 18) {
        return false;
      }
      if (tile == 20 || 22 <= tile && tile <= 28) {
        return false;
      }
      continue;
    }

    if (597 <= encode && encode <= 956) {
      std::uint_fast8_t const seat_ = (encode - 597) / 90;
      if (seat_ == (seat + 1) % 4) {
        return false;
      }
      continue;
    }

    if (957 <= encode && encode <= 1436) {
      std::uint_fast8_t const seat_ = (encode - 957) / 120;
      std::uint_fast8_t const relseat = (encode - 957 - seat_ * 120) / 40;
      if ((seat_ + relseat + 1) % 4 == seat) {
        return false;
      }
      continue;
    }

    if (1437 <= encode && encode <= 1880) {
      std::uint_fast8_t const seat_ = (encode - 1437) / 111;
      std::uint_fast8_t const relseat = (encode - 1437 - seat_ * 111) / 37;
      if ((seat_ + relseat + 1) % 4 == seat) {
        return false;
      }
      continue;
    }
  }

  return true;
}

bool RoundState::checkSifengLianda() const
{
  if (python::len(progression_) != 5u) {
    return false;
  }
  KANACHAN_ASSERT((python::extract<long>(progression_[0]) == 0));

  std::array<std::uint_fast8_t, 4u> tiles = {
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max()
  };
  for (std::uint_fast8_t i = 1u; i <= 4u; ++i) {
    python::object o = progression_[i];
    KANACHAN_ASSERT((!o.is_none()));
    python::extract<long> e(o);
    KANACHAN_ASSERT((e.check()));
    KANACHAN_ASSERT((0 <= e() && e() <= 2164));
    std::uint_fast16_t const encode = e();
    if (encode < 5u || 596u < encode) {
      return false;
    }
    long const p = encode - 5u;
    std::uint_fast8_t const seat = p / 148;
    std::uint_fast8_t const tile = (p - seat * 148) / 4;
    tiles[i - 1u] = tile;
  }

  for (std::uint_fast8_t i = 30u; i < 34u; ++i) {
    bool const result = std::all_of(
      tiles.cbegin(), tiles.cend(),
      [i](std::uint_fast8_t const tile){ return tile == i; });
    if (result) {
      return true;
    }
  }
  return false;
}

bool RoundState::checkSigangSanle() const
{
  std::uint_fast8_t num_ganzi = 0u;
  bool flag = false;
  for (Kanachan::Shoupai const &s : shoupai_list_) {
    std::uint_fast8_t const n = s.getNumGangzi();
    if (num_ganzi >= 1u && n >= 1u) {
      // 複数のプレイヤが槓をしている．
      flag = true;
    }
    num_ganzi += n;
  }
  KANACHAN_ASSERT((num_ganzi <= 4u));
  return num_ganzi == 4u && flag;
}

bool RoundState::checkSijiaLizhi() const
{
  std::uint_fast8_t n = 0u;
  for (std::uint_fast8_t lizhi : lizhi_list_) {
    if (lizhi >= 1u) {
      ++n;
    }
  }
  return n == 4u;
}

std::pair<std::uint_fast8_t, std::uint_fast8_t> RoundState::getLastDapai_() const
{
  KANACHAN_ASSERT((python::len(progression_) >= 1u));
  python::object o = progression_[-1];
  python::extract<long> e(o);
  KANACHAN_ASSERT((e.check()));
  KANACHAN_ASSERT((e() > 0));
  std::uint_fast16_t const encode = e();

  KANACHAN_ASSERT((encode >= 5u));
  if (/*5u <= encode && */encode <= 596u) {
    std::uint_fast8_t const seat = (encode - 5u) / 148u;
    std::uint_fast8_t const tile = (encode - 5u - seat * 148u) / 4u;
    return { seat, tile };
  }
  KANACHAN_ASSERT((encode >= 1881u));
  if (/*1881u <= encode && */encode <= 2016u) {
    // 暗槓．槍槓の際には打牌とみなす．
    std::uint_fast8_t const seat = (encode - 1881u) / 34u;
    std::uint_fast8_t const tile = (encode - 1881u - seat * 34u);
    if (0u <= tile && tile <= 8u) {
      return { seat, tile + 1u };
    }
    if (9u <= tile && tile <= 17u) {
      return { seat, tile + 2u };
    }
    if (18u <= tile && tile < 34u) {
      return { seat, tile + 3u };
    }
    KANACHAN_THROW<std::logic_error>(_1) << tile;
  }
  if (/*2017u <= encode && */encode <= 2164u) {
    // 加槓．槍槓の際には打牌とみなす．
    std::uint_fast8_t const seat = (encode - 2017u) / 37u;
    std::uint_fast8_t const tile = (encode - 2017u - seat * 37u);
    return { seat, tile };
  }
  KANACHAN_THROW<std::logic_error>(_1) << encode;
}

std::uint_fast8_t RoundState::drawLingshangPai_()
{
  KANACHAN_ASSERT((getNumLeftTiles() >= 1u));
  KANACHAN_ASSERT((lingshang_zimo_count_ < 4u));
  KANACHAN_ASSERT((gang_dora_count_ <= 4u));
  KANACHAN_ASSERT((lizhi_delayed_ == 0u));
  KANACHAN_ASSERT((!angang_dora_delayed_));
  KANACHAN_ASSERT((!first_zimo_[0u]));
  KANACHAN_ASSERT((!first_zimo_[1u]));
  KANACHAN_ASSERT((!first_zimo_[2u]));
  KANACHAN_ASSERT((!first_zimo_[3u]));
  KANACHAN_ASSERT((!yifa_[0u]));
  KANACHAN_ASSERT((!yifa_[1u]));
  KANACHAN_ASSERT((!yifa_[2u]));
  KANACHAN_ASSERT((!yifa_[3u]));
  KANACHAN_ASSERT((lingshang_kaihua_delayed_));
  KANACHAN_ASSERT((!qianggang_delayed_));

  // 牌山の王牌および嶺上牌は各幢の上下が逆になっている．
  std::uint_fast8_t const tile = paishan_[135 - lingshang_zimo_count_];
  ++lingshang_zimo_count_;
  return tile;
}

python::list RoundState::constructFeatures_(
  std::uint_fast8_t seat, std::uint_fast8_t const zimo_tile) const
{
  python::list sparse_features;

  long const room = game_state_.getRoom();
  sparse_features.append(room);

  long const game_style = game_state_.isDongfengZhan() ? 5 : 6;
  sparse_features.append(game_style);

  sparse_features.append(7 + seat);

  long const chang = 11 + getChang();
  sparse_features.append(chang);

  long const ju = 14 + getJu();
  sparse_features.append(ju);

  long const dora_indicator = getDoraIndicator(0u);
  KANACHAN_ASSERT((dora_indicator != std::numeric_limits<std::uint_fast8_t>::max()));
  sparse_features.append(18 + dora_indicator);

  for (std::uint_fast8_t i = 1u; i <= 4u; ++i) {
    long const gang_dora_indicator = getDoraIndicator(i);
    if (gang_dora_indicator == std::numeric_limits<std::uint_fast8_t>::max()) {
      break;
    }
    sparse_features.append(18 + i * 37 + gang_dora_indicator);
  }

  long const num_left_tiles = getNumLeftTiles();
  KANACHAN_ASSERT((0 <= num_left_tiles && num_left_tiles < 70));
  sparse_features.append(203 + num_left_tiles);

  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    long const player_grade = getPlayerGrade((seat + i) % 4u);
    sparse_features.append(273 + i * 20 + player_grade);

    long const player_ranking = getPlayerRanking((seat + i) % 4u);
    sparse_features.append(289 + i * 20 + player_ranking);
  }

  Kanachan::Shoupai const &shoupai = shoupai_list_[seat];
  shoupai.appendToFeatures(sparse_features);

  if (zimo_tile != std::numeric_limits<std::uint_fast8_t>::max()) {
    sparse_features.append(489 + zimo_tile);
  }

  python::list numeric_features;

  long const ben_chang = getBenChang();
  numeric_features.append(ben_chang);

  long const lizhi_deposits = getNumLizhiDeposits();
  numeric_features.append(lizhi_deposits);

  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    long const player_score = getPlayerScore((seat + i) % 4u);
    numeric_features.append(player_score);
  }

  python::list features;
  features.append(sparse_features);
  features.append(numeric_features);
  return features;
}

std::uint_fast16_t RoundState::selectAction_(
  std::uint_fast8_t const seat, python::list features) const
{
  return game_state_.selectAction(seat, features);
}

long RoundState::encodeToolConfig_(
  std::uint_fast8_t const seat, bool const rong) const
{
  KANACHAN_ASSERT((seat < 4u));

  long tool_config = 0u;
  if (!rong) {
    tool_config |= 1u << 0u; // Zi Mo (自摸)
  }
  if (lizhi_list_[seat] == 1u) {
    tool_config |= 1u << 1u; // Li Zhi (立直)
  }
  if (yifa_[seat]) {
    tool_config |= 1u << 2u; // Yi Fa (一発)
  }
  if (lingshang_kaihua_delayed_) {
    KANACHAN_ASSERT((!rong));
    tool_config |= 1u << 3u; // Ling Shang Kai Hua (嶺上開花)
  }
  if (qianggang_delayed_) {
    KANACHAN_ASSERT((rong));
    tool_config |= 1u << 4u; // Qiang Gang (槍槓)
  }
  if (getNumLeftTiles() == 0u && !rong && !lingshang_kaihua_delayed_) {
    // 最終自摸牌が嶺上牌である場合は海底撈月は成立しない．
    tool_config |= 1u << 5u; // Hai Di Lao Yue (海底撈月)
  }
  if (getNumLeftTiles() == 0u && rong) {
    tool_config |= 1u << 6u; // He Di Lao Yu (河底撈魚)
  }
  if (lizhi_list_[seat] == 2u) {
    tool_config |= 1u << 7u; // Double Li Zhi (ダブル立直)
  }
  if (!rong && first_zimo_[seat]) {
    if (seat == getJu()) {
      tool_config |= 1u << 8u; // Tian Hu (天和)
    }
    else {
      tool_config |= 1u << 9u; // Di Hu (地和)
    }
  }
  tool_config |= 1u << (10u + (seat + 4u - getJu()) % 4u); // Men Feng Pai (門風牌, 自風)
  tool_config |= 1u << (14u + getChang()); // Quan Feng Pai (圏風牌, 場風)

  return tool_config;
}

python::list
RoundState::constructDoraIndicators_(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));

  bool const lizhi = lizhi_list_[seat];
  python::list dora_indicators;

  auto append = [dora_indicators](std::uint_fast8_t const tile) mutable -> void {
    KANACHAN_ASSERT((tile < 37u));

    if (tile == 0u) {
      dora_indicators.append(16);
    }
    else if (1u <= tile && tile <= 9u) {
      for (std::uint_fast8_t i = (tile == 5u ? 1u : 0u); i < 4u; ++i) {
        long const encode = (tile - 1u) * 4u + i;
        long const count = dora_indicators.count(encode);
        if (count >= 1) {
          KANACHAN_ASSERT((count == 1));
          KANACHAN_ASSERT((i != 3u));
          continue;
        }
        dora_indicators.append(encode);
        break;
      }
    }
    else if (tile == 10u) {
      dora_indicators.append(52);
    }
    else if (11u <= tile && tile <= 19u) {
      for (std::uint_fast8_t i = (tile == 15u ? 1u : 0u); i < 4u; ++i) {
        long const encode = (tile - 2u) * 4u + i;
        long const count = dora_indicators.count(encode);
        if (count >= 1) {
          KANACHAN_ASSERT((count == 1));
          KANACHAN_ASSERT((i != 3u));
          continue;
        }
        dora_indicators.append(encode);
        break;
      }
    }
    else if (tile == 20u) {
      dora_indicators.append(88);
    }
    else if (21u <= tile && tile < 37u) {
      for (std::uint_fast8_t i = (tile == 25u ? 1u : 0u); i < 4u; ++i) {
        long const encode = (tile - 3u) * 4u + i;
        long const count = dora_indicators.count(encode);
        if (count >= 1) {
          KANACHAN_ASSERT((count == 1));
          KANACHAN_ASSERT((i != 3u));
          continue;
        }
        dora_indicators.append(encode);
        break;
      }
    }
  };

  // 牌山の王牌および嶺上牌は各幢の上下が逆になっている．
  for (std::uint_fast8_t i = 0u; i <= gang_dora_count_; ++i) {
    std::uint_fast8_t const dora_indicator = paishan_[131u - i * 2];
    append(dora_indicator);
    if (lizhi) {
      std::uint_fast8_t const lidora_indicator = paishan_[130u - i * 2];
      append(lidora_indicator);
    }
  }

  dora_indicators.attr("sort")();
  return dora_indicators;
}

std::pair<std::uint_fast8_t, std::uint_fast8_t>
RoundState::checkDaSanyuanPao_() const
{
  std::uint_fast8_t to = std::numeric_limits<std::uint_fast8_t>::max();
  std::uint_fast8_t count = 0u;
  for (long i = 0; i < python::len(progression_); ++i) {
    python::object o = progression_[i];
    python::extract<long> e(o);
    KANACHAN_ASSERT((e.check()));
    KANACHAN_ASSERT((0 <= e()));
    KANACHAN_ASSERT((e() <= 2164));
    std::uint_fast16_t const encode = e();

    if (/*0u <= encode && */encode <= 956u) {
      continue;
    }
    if (/*957u <= encode && */encode <= 1436u) {
      // ポン
      std::uint_fast8_t const seat = (encode - 957u) / 120u;
      std::uint_fast8_t const relseat = (encode - 957u - seat * 120u) / 40u;
      std::uint_fast8_t const peng
        = (encode - 957u - seat * 120u - relseat * 40u);
      if (peng <= 36u) {
        continue;
      }
      KANACHAN_ASSERT((peng <= 39u));
      if (to == std::numeric_limits<std::uint_fast8_t>::max()) {
        to = seat;
      }
      else if (to != seat) {
        return {
          std::numeric_limits<std::uint_fast8_t>::max(),
          std::numeric_limits<std::uint_fast8_t>::max()
        };
      }
      ++count;
      if (count == 3u) {
        return { (seat + relseat + 1u) % 4u, to };
      }
      continue;
    }
    if (/*1437u <= encode && */encode <= 1880u) {
      // 大明槓
      std::uint_fast8_t const seat = (encode - 1437u) / 111u;
      std::uint_fast8_t const relseat = (encode - 1437u - seat * 111u) / 37u;
      std::uint_fast8_t const tile = (encode - 1437u - seat * 111u - relseat * 37u);
      if (tile < 34u) {
        continue;
      }
      KANACHAN_ASSERT((tile < 37u));
      if (to == std::numeric_limits<std::uint_fast8_t>::max()) {
        to = seat;
      }
      else if (to != seat) {
        return {
          std::numeric_limits<std::uint_fast8_t>::max(),
          std::numeric_limits<std::uint_fast8_t>::max()
        };
      }
      ++count;
      if (count == 3u) {
        return { (seat + relseat + 1u) % 4u, to };
      }
      continue;
    }
    if (/*1881u <= encode && */encode <= 2016u) {
      // 暗槓
      std::uint_fast8_t const seat = (encode - 1881u) / 34u;
      std::uint_fast8_t const tile = (encode - 1881u - seat * 34u);
      if (tile < 31u) {
        continue;
      }
      KANACHAN_ASSERT((tile < 34u));
      if (to == std::numeric_limits<std::uint_fast8_t>::max()) {
        to = seat;
      }
      else if (to != seat) {
        return {
          std::numeric_limits<std::uint_fast8_t>::max(),
          std::numeric_limits<std::uint_fast8_t>::max()
        };
      }
      ++count;
      continue;
    }
    if (2017u <= encode && encode <= 2164u) {
      // 加槓
      continue;
    }
    KANACHAN_THROW<std::logic_error>(_1) << encode << ": A logic error.";
  }
  KANACHAN_ASSERT((count < 3u));
  return {
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max()
  };
}

std::pair<std::uint_fast8_t, std::uint_fast8_t>
RoundState::checkDaSixiPao_() const
{
  std::uint_fast8_t to = -1;
  std::uint_fast8_t count = 0u;
  for (long i = 0; i < python::len(progression_); ++i) {
    python::object o = progression_[i];
    python::extract<long> e(o);
    KANACHAN_ASSERT((e.check()));
    KANACHAN_ASSERT((0 <= e()));
    KANACHAN_ASSERT((e() <= 2164));
    std::uint_fast16_t const encode = e();

    if (/*0u <= encode && */encode <= 956u) {
      continue;
    }
    if (/*957u <= encode && */encode <= 1436u) {
      // ポン
      std::uint_fast8_t const seat = (encode - 957u) / 120u;
      std::uint_fast8_t const relseat = (encode - 957u - seat * 120u) / 40u;
      std::uint_fast8_t const peng
        = (encode - 957u - seat * 120u - relseat * 40u);
      if (peng < 33u || 36 < peng) {
        // 風牌以外のポン
        continue;
      }
      if (to == static_cast<std::uint_fast8_t>(-1)) {
        to = seat;
      }
      else if (to != seat) {
        return {
          static_cast<std::uint_fast8_t>(-1), static_cast<std::uint_fast8_t>(-1)
        };
      }
      ++count;
      if (count == 4u) {
        return { (seat + relseat + 1u) % 4u, to };
      }
      continue;
    }
    if (/*1437u <= encode && */encode <= 1880u) {
      // 大明槓
      std::uint_fast8_t const seat = (encode - 1437u) / 111u;
      std::uint_fast8_t const relseat = (encode - 1437u - seat * 111u) / 37u;
      std::uint_fast8_t const tile = (encode - 1437u - seat * 111u - relseat * 37u);
      if (tile < 30u || 33u < tile) {
        // 風牌以外の大明槓．
        continue;
      }
      if (to == static_cast<std::uint_fast8_t>(-1)) {
        to = seat;
      }
      else if (to != seat) {
        return {
          static_cast<std::uint_fast8_t>(-1), static_cast<std::uint_fast8_t>(-1)
        };
      }
      ++count;
      if (count == 4u) {
        return { (seat + relseat + 1u) % 4u, to };
      }
      continue;
    }
    if (/*1881u <= encode && */encode <= 2016u) {
      // 暗槓
      std::uint_fast8_t const seat = (encode - 1881u) / 34u;
      std::uint_fast8_t const tile = (encode - 1881u - seat * 34u);
      if (tile < 27u || 30u < tile) {
        // 風牌以外の暗槓．
        continue;
      }
      if (to == static_cast<std::uint_fast8_t>(-1)) {
        to = seat;
      }
      else if (to != seat) {
        return {
          static_cast<std::uint_fast8_t>(-1), static_cast<std::uint_fast8_t>(-1)
        };
      }
      ++count;
      continue;
    }
    if (2017u <= encode && encode <= 2164u) {
      // 加槓
      continue;
    }
    KANACHAN_THROW<std::logic_error>(_1) << encode << ": A logic error.";
  }
  KANACHAN_ASSERT((count < 3u));
  return {
    static_cast<std::uint_fast8_t>(-1), static_cast<std::uint_fast8_t>(-1)
  };
}

std::pair<std::uint_fast8_t, std::uint_fast8_t>
RoundState::calculateHand_(
  std::uint_fast8_t const seat, std::uint_fast8_t const hupai,
  long const config) const
{
  KANACHAN_ASSERT((seat < 4u));
  KANACHAN_ASSERT((config >= 0));

  Kanachan::Shoupai const &shoupai = shoupai_list_[seat];
  python::list dora_indicators = constructDoraIndicators_(seat);
  return shoupai.calculateHand(hupai, dora_indicators, config);
}

void RoundState::settleLizhiDeposits_()
{
  std::uint_fast8_t seat = std::numeric_limits<std::uint_fast8_t>::max();
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    if (getPlayerRanking(i) == 0u) {
      seat = i;
      break;
    }
  }
  KANACHAN_ASSERT((seat < 4u));

  game_state_.addPlayerScore(seat, 1000 * getNumLizhiDeposits());
}

std::pair<std::uint_fast16_t, std::uint_fast8_t> RoundState::onZimo()
{
  KANACHAN_ASSERT((getNumLeftTiles() >= 1u));
  KANACHAN_ASSERT((lingshang_zimo_count_ <= 4u));
  KANACHAN_ASSERT((gang_dora_count_ <= 4u));
  KANACHAN_ASSERT((seat_ < 4u));
  KANACHAN_ASSERT((lizhi_delayed_ == 0u));
  KANACHAN_ASSERT((!qianggang_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!rong_delayed_[i]));
  }

  std::uint_fast8_t zimo_tile = std::numeric_limits<std::uint_fast8_t>::max();
  if (angang_dora_delayed_) {
    // 暗槓直後の嶺上自摸
    KANACHAN_ASSERT((lingshang_zimo_count_ < 4u));
    KANACHAN_ASSERT((gang_dora_count_ < 4u));
    KANACHAN_ASSERT((!minggang_dora_delayed_));
    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      KANACHAN_ASSERT((!first_zimo_[i]));
      KANACHAN_ASSERT((!yifa_[i]));
    }
    KANACHAN_ASSERT((lingshang_kaihua_delayed_));
    ++gang_dora_count_;
    angang_dora_delayed_ = false;
    zimo_tile = drawLingshangPai_();
  }
  else if (minggang_dora_delayed_) {
    // 明槓直後の嶺上自摸
    KANACHAN_ASSERT((lingshang_zimo_count_ < 4u));
    KANACHAN_ASSERT((gang_dora_count_ < 4u));
    KANACHAN_ASSERT((!angang_dora_delayed_));
    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      KANACHAN_ASSERT((!first_zimo_[i]));
      KANACHAN_ASSERT((!yifa_[i]));
    }
    KANACHAN_ASSERT((lingshang_kaihua_delayed_));
    zimo_tile = drawLingshangPai_();
  }
  else {
    KANACHAN_ASSERT((!angang_dora_delayed_));
    KANACHAN_ASSERT((!minggang_dora_delayed_));
    zimo_tile = paishan_[zimo_index_++];
  }

  Kanachan::Shoupai &shoupai = shoupai_list_[seat_];

  python::list features = constructFeatures_(seat_, zimo_tile);
  features.append(progression_);

  bool const first_zimo = first_zimo_[seat_];
  bool const haidi = (getNumLeftTiles() == 0u);
  bool const lizhi_prohibited
    = (getPlayerScore(seat_) < 1000 || getNumLeftTiles() < 4u);
  bool const gang_prohibited = (lingshang_zimo_count_ == 4u || haidi);
  long const tool_config = encodeToolConfig_(seat_, /*rong = */false);
  python::list candidates = shoupai.getCandidatesOnZimo(
    zimo_tile, first_zimo, lizhi_prohibited, gang_prohibited, tool_config);
  features.append(candidates);
  std::uint_fast16_t action = std::numeric_limits<std::uint_fast16_t>::max();
  if (python::len(candidates) >= 1) {
    KANACHAN_ASSERT((python::len(candidates) >= 2));
    action = selectAction_(seat_, features);
  }

  std::uint_fast8_t tile = std::numeric_limits<std::uint_fast8_t>::max();

  if (action <= 147u) {
    // Da Pai (打牌)
    tile = action / 4u;
    bool const moqi = ((action - tile * 4u) / 2u == 1u);
    bool const lizhi = (action - tile * 4u - moqi * 2u == 1u);
    KANACHAN_ASSERT((!moqi || tile == zimo_tile));
    if (lizhi) {
      if (first_zimo_[seat_]) {
        lizhi_delayed_ = 2u;
      }
      else {
        lizhi_delayed_ = 1u;
      }
      yifa_[seat_] = true;
    }
    else {
      yifa_[seat_] = false;
    }
    first_zimo_[seat_] = false;
    lingshang_kaihua_delayed_ = false;
    shoupai.onPostZimo(zimo_tile, tile, lizhi_list_[seat_] >= 1u);
  }

  if (action == std::numeric_limits<std::uint_fast16_t>::max()) {
    // Da Pai after Li Zhi (立直後の打牌)
    KANACHAN_ASSERT((lizhi_list_[seat_] >= 1u));
    KANACHAN_ASSERT((!first_zimo_[seat_]));
    yifa_[seat_] = false;
    lingshang_kaihua_delayed_ = false;
    tile = zimo_tile;
  }

  if (148u <= action && action <= 181u) {
    // An Gang (暗槓)
    KANACHAN_ASSERT((lizhi_delayed_ == 0u));
    // 一発の消失は槓の成立後．つまり槍槓により槓が不成立の場合，
    // 一発は消失せず，槍槓と複合しうる．ただし，暗槓に対する槍槓は
    // 国士無双に限られるため，一発の消失のタイミングが問題になることは無い．
    // 第一自摸の判定も同様だが，こちらもタイミングが問題になることは無い．
    lingshang_kaihua_delayed_ = false;
    tile = zimo_tile;
  }

  if (182u <= action && action <= 218u) {
    // Jiang Gang (加槓)
    KANACHAN_ASSERT((lizhi_delayed_ == 0u));
    // 一発の消失は槓の成立後．つまり槍槓により槓が不成立の場合，
    // 一発は消失せず，槍槓と複合しうる．第一自摸の判定も同様だが，
    // こちらはタイミングが問題になることは無い．
    lingshang_kaihua_delayed_ = false;
    tile = zimo_tile;
  }

  if (action == 219u) {
    // Zi Mo Hu (自摸和)
    KANACHAN_ASSERT((lizhi_delayed_ == 0u));
    KANACHAN_ASSERT((!angang_dora_delayed_));
    tile = zimo_tile;
  }

  if (action == 220u) {
    // Jiu Zhong Jiu Pai (九種九牌)
    KANACHAN_ASSERT((lizhi_delayed_ == 0u));
    KANACHAN_ASSERT((!angang_dora_delayed_));
    KANACHAN_ASSERT((!minggang_dora_delayed_));
    KANACHAN_ASSERT((first_zimo_[seat_]));
    KANACHAN_ASSERT((!yifa_[seat_]));
    KANACHAN_ASSERT((!lingshang_kaihua_delayed_));
  }

  if (221u <= action && action != std::numeric_limits<std::uint_fast16_t>::max()) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << action << ": An invalid action on zimo.";
  }

  return { action, tile };
}

std::pair<std::uint_fast8_t, std::uint_fast16_t>
RoundState::onDapai(
  std::uint_fast8_t const tile, bool const moqi, bool const lizhi)
{
  KANACHAN_ASSERT((zimo_index_ <= 122u - lingshang_zimo_count_));
  KANACHAN_ASSERT((lingshang_zimo_count_ <= 4u));
  KANACHAN_ASSERT((gang_dora_count_ <= 4u));
  KANACHAN_ASSERT((seat_ < 4u));
  KANACHAN_ASSERT((!angang_dora_delayed_));
  KANACHAN_ASSERT((!lingshang_kaihua_delayed_));
  KANACHAN_ASSERT((!qianggang_delayed_));
  KANACHAN_ASSERT((!rong_delayed_[0u]));
  KANACHAN_ASSERT((!rong_delayed_[1u]));
  KANACHAN_ASSERT((!rong_delayed_[2u]));
  KANACHAN_ASSERT((!rong_delayed_[3u]));

  if (minggang_dora_delayed_) {
    KANACHAN_ASSERT((lingshang_zimo_count_ == gang_dora_count_ + 1u));
    KANACHAN_ASSERT((lizhi_delayed_ == 0u));
    KANACHAN_ASSERT((!first_zimo_[0u]));
    KANACHAN_ASSERT((!first_zimo_[1u]));
    KANACHAN_ASSERT((!first_zimo_[2u]));
    KANACHAN_ASSERT((!first_zimo_[3u]));
    KANACHAN_ASSERT((!yifa_[0u]));
    KANACHAN_ASSERT((!yifa_[1u]));
    KANACHAN_ASSERT((!yifa_[2u]));
    KANACHAN_ASSERT((!yifa_[3u]));
    ++gang_dora_count_;
  }
  minggang_dora_delayed_ = false;

  std::uint_fast8_t const moqi_ = moqi ? 1u : 0u;
  std::uint_fast8_t const lizhi_ = lizhi ? 1u : 0u;
  if (lizhi) {
    KANACHAN_ASSERT((lizhi_delayed_ >= 1u));
  }
  progression_.append(5 + seat_ * 148 + tile * 4u + moqi_ * 2u + lizhi_);

  std::array<std::uint_fast16_t, 3u> actions = {
    std::numeric_limits<std::uint_fast16_t>::max(),
    std::numeric_limits<std::uint_fast16_t>::max(),
    std::numeric_limits<std::uint_fast16_t>::max()
  };
  for (std::uint_fast8_t i = 0; i < 3u; ++i) {
    std::uint_fast8_t const calling_seat = (seat_ + i + 1) % 4u;
    std::uint_fast8_t const zimo_tile = -1;
    python::list features = constructFeatures_(calling_seat, zimo_tile);
    features.append(progression_);
    Kanachan::Shoupai const &shoupai = shoupai_list_[calling_seat];
    long const tool_config = encodeToolConfig_(calling_seat, /*rong = */true);
    python::list candidates = shoupai.getCandidatesOnDapai(
      2u - i, tile, lingshang_zimo_count_ == 4u, tool_config);
    KANACHAN_ASSERT((python::len(candidates) != 1));
    if (checkSigangSanle()) {
      // 4つ目の槓に対する打牌の場合，その打牌に対する他家の選択肢から
      // スキップと栄和以外を除外しなければならない．
      for (long j = 0; j < python::len(candidates);) {
        python::object o = candidates[j];
        python::extract<long> e(o);
        KANACHAN_ASSERT((e.check()));
        KANACHAN_ASSERT((0 <= e() && e() <= 545));
        std::uint_fast16_t const candidate = e();
        if (candidate != 221u && candidate < 543u) {
          candidates.attr("pop")(j);
          continue;
        }
        ++j;
      }
    }
    if (python::len(candidates) == 1) {
      python::object o = candidates[0];
      python::extract<long> e(o);
      KANACHAN_ASSERT((e.check()));
      KANACHAN_ASSERT((e() == 221));
      continue;
    }
    if (python::len(candidates) == 0) {
      continue;
    }
    features.append(candidates);
    try {
      std::uint_fast16_t const action = selectAction_(calling_seat, features);
      actions[i] = action;
    }
    catch (std::runtime_error const &e) {
      if (checkSigangSanle()) {
        // TestModel を使用したテストを行っている場合で，四槓散了が成立する時，
        // 直前の `selectAction_` で次局の親の打牌の選択肢を読んでしまう．
        // これにより `selectAction_` から例外が送出されるが，テストとしては
        // 正常な動作であるため，実行を続けなければならない．
        break;
      }
      throw;
    }
  }

  for (;;) {
    // 栄和が選択されたかどうかをチェックする．
    std::uint_fast8_t i = 0u;
    for (; i < 3u; ++i) {
      if (543u <= actions[i] && actions[i] <= 545u) {
        break;
      }
    }
    if (i == 3u) {
      break;
    }

    // 栄和が選択された．
    for (std::uint_fast8_t j = 0u; j < 3u; ++j) {
      if (actions[j] == 221) {
        continue;
      }
      if (222u <= actions[j] && actions[j] <= 311u) {
        // Chi that is being canceled by rong.
        continue;
      }
      if (312u <= actions[j] && actions[j] <= 431u) {
        // Peng that is being canceled by rong.
        continue;
      }
      if (432u <= actions[j] && actions[j] <= 542u) {
        // Da ming gang that is being canceled by rong.
        continue;
      }
      if (543u <= actions[j] && actions[j] <= 545u) {
        std::uint_fast8_t const calling_seat = (seat_ + j + 1u) % 4u;
        rong_delayed_[calling_seat] = true;
        continue;
      }
      if (actions[j] == std::numeric_limits<std::uint_fast16_t>::max()) {
        continue;
      }
      KANACHAN_THROW<std::runtime_error>(_1)
        << static_cast<unsigned>(j) << ": " << actions[j]
        << ": An invalid rong action.";
    }

    seat_ = std::numeric_limits<std::uint_fast8_t>::max();
    return { /*seat = */std::numeric_limits<std::uint_fast8_t>::max(), 543u };
  }

  if (checkSigangSanle()) {
    // Si Gang San Le (四槓散了)
    // 四槓散了の成立は打牌時であり，なおかつ，当該打牌に対する栄和の成立よりも
    // 後である．例として 200718-29d44d71-07f4-4ff2-b000-80d6ed362b17
    // 東3局1本場を参照のこと．
    return { seat_, std::numeric_limits<std::uint_fast16_t>::max() };
  }

  if (lizhi_delayed_ >= 1u) {
    // 栄和されなかったので立直が成立する．
    lizhi_list_[seat_] = lizhi_delayed_;
    yifa_[seat_] = true;
    game_state_.onSuccessfulLizhi(seat_);
    lizhi_delayed_ = 0u;
  }

  if (checkSifengLianda() || checkSijiaLizhi()) {
    // Si Feng Liand or Si Jia Li Zhi (四風連打または四家立直)
    return { seat_, std::numeric_limits<std::uint_fast16_t>::max() };
  }

  for (;;) {
    // ポンまたは大明槓が選択されたかどうかをチェックする．
    std::uint_fast8_t i = 0u;
    for (; i < 3u; ++i) {
      if (312u <= actions[i] && actions[i] <= 431u) {
        break;
      }
      if (432u <= actions[i] && actions[i] <= 542u) {
        break;
      }
    }
    if (i == 3u) {
      break;
    }

    // ポンまたは大明槓が選択された．
    for (std::uint_fast8_t j = 0u; j < 3u; ++j) {
      if (j == i) {
        continue;
      }
      if (actions[j] == 221) {
        continue;
      }
      if (222u <= actions[j] && actions[j] <= 311u) {
        // Chi that is being canceled by peng or da ming gang.
        continue;
      }
      if (actions[j] == std::numeric_limits<std::uint_fast16_t>::max()) {
        continue;
      }
      KANACHAN_THROW<std::runtime_error>(_1)
        << static_cast<unsigned>(j) << ": " << actions[j]
        << ": An invalid peng or da ming gang action.";
    }

    first_zimo_ = { false, false, false, false };
    yifa_ = { false, false, false, false };
    std::uint_fast8_t const dapai_seat = seat_;
    seat_ = (seat_ + i + 1u) % 4u;
    return { dapai_seat, actions[i] };
  }

  for (;;) {
    // チーが選択されたかどうかチェックする．
    std::uint_fast8_t i = 0u;
    for (; i < 3u; ++i) {
      if (222u <= actions[i] && actions[i] <= 311u) {
        break;
      }
    }
    if (i == 3u) {
      break;
    }

    if (i != 0u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << static_cast<unsigned>(i) << ": An invalid chi.";
    }

    // チーが選択された．
    for (std::uint_fast8_t j = 1u; j < 3u; ++j) {
      if (actions[j] == 221u) {
        continue;
      }
      if (actions[j] == std::numeric_limits<std::uint_fast16_t>::max()) {
        continue;
      }
      KANACHAN_THROW<std::runtime_error>(_1)
        << static_cast<unsigned>(j) << ": " << actions[j]
        << ": An invalid chi action.";
    }

    first_zimo_ = { false, false, false, false };
    yifa_ = { false, false, false, false };
    std::uint_fast8_t const dapai_seat = seat_;
    seat_ = (seat_ + 1u) % 4u;
    return { dapai_seat, actions[i] };
  }

  // 選択肢がでなかった，もしくは全ての選択肢でスキップが選択された．
  for (std::uint_fast8_t i = 0; i < 3u; ++i) {
    if (actions[i] == 221u) {
      continue;
    }
    if (actions[i] == std::numeric_limits<std::uint_fast16_t>::max()) {
      continue;
    }
    KANACHAN_THROW<std::runtime_error>(_1)
      << static_cast<unsigned>(i) << ": " << actions[i]
      << ": An invalid action.";
  }
  seat_ = (seat_ + 1u) % 4u;
  return { std::numeric_limits<std::uint_fast8_t>::max(), 221u };
}

std::uint_fast16_t RoundState::onChi(std::uint_fast8_t const encode)
{
  KANACHAN_ASSERT((encode < 90u));

  KANACHAN_ASSERT((getNumLeftTiles() >= 1u));
  KANACHAN_ASSERT((lingshang_zimo_count_ <= 4u));
  KANACHAN_ASSERT((gang_dora_count_ <= 4u));
  KANACHAN_ASSERT((seat_ < 4u));
  KANACHAN_ASSERT((lizhi_list_[seat_] == 0u));
  KANACHAN_ASSERT((lizhi_delayed_ == 0u));
  KANACHAN_ASSERT((!angang_dora_delayed_));
  KANACHAN_ASSERT((!minggang_dora_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!first_zimo_[i]));
  }
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!yifa_[i]));
  }
  KANACHAN_ASSERT((!lingshang_kaihua_delayed_));
  KANACHAN_ASSERT((!qianggang_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!rong_delayed_[i]));
  }

  auto const [dapai_seat, dapai] = getLastDapai_();

  Kanachan::Shoupai &shoupai = shoupai_list_[seat_];
  shoupai.onChi(encode);

  std::uint_fast8_t const zimo_tile = std::numeric_limits<std::uint_fast8_t>::max();
  python::list features = constructFeatures_(seat_, zimo_tile);

  progression_.append(597 + seat_ * 90 + encode);
  features.append(progression_);

  python::list candidates = shoupai.getCandidatesOnChiPeng();
  features.append(candidates);

  std::uint_fast16_t const action = selectAction_(seat_, features);
  if (action >= 148u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << action << ": An invalid dapai action after chi.";
  }

  {
    std::uint_fast8_t const dapai = action / 4u;
    shoupai.onPostChiPeng(dapai);
  }

  return action;
}

std::uint_fast16_t RoundState::onPeng(std::uint_fast8_t const encode)
{
  KANACHAN_ASSERT((encode < 40u));

  KANACHAN_ASSERT((getNumLeftTiles() >= 1u));
  KANACHAN_ASSERT((lingshang_zimo_count_ <= 4u));
  KANACHAN_ASSERT((gang_dora_count_ <= 4u));
  KANACHAN_ASSERT((seat_ < 4u));
  KANACHAN_ASSERT((lizhi_list_[seat_] == 0u));
  KANACHAN_ASSERT((lizhi_delayed_ == 0u));
  KANACHAN_ASSERT((!angang_dora_delayed_));
  KANACHAN_ASSERT((!minggang_dora_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!first_zimo_[i]));
  }
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!yifa_[i]));
  }
  KANACHAN_ASSERT((!lingshang_kaihua_delayed_));
  KANACHAN_ASSERT((!qianggang_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!rong_delayed_[i]));
  }

  auto const [dapai_seat, dapai] = getLastDapai_();
  std::uint_fast8_t const relseat = (dapai_seat + 4u - seat_) % 4u - 1u;

  Kanachan::Shoupai &shoupai = shoupai_list_[seat_];
  shoupai.onPeng(relseat, encode);

  std::uint_fast8_t const zimo_tile = std::numeric_limits<std::uint_fast8_t>::max();
  python::list features = constructFeatures_(seat_, zimo_tile);

  progression_.append(957 + seat_ * 120 + relseat * 40 + encode);
  features.append(progression_);

  python::list candidates = shoupai.getCandidatesOnChiPeng();
  features.append(candidates);

  std::uint_fast16_t const action = selectAction_(seat_, features);
  if (action >= 148u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << action << ": An invalid dapai action after peng.";
  }

  {
    std::uint_fast8_t const dapai = action / 4u;
    shoupai.onPostChiPeng(dapai);
  }

  return action;
}

void RoundState::onDaminggang()
{
  KANACHAN_ASSERT((getNumLeftTiles() >= 1u));
  KANACHAN_ASSERT((lingshang_zimo_count_ < 4u));
  KANACHAN_ASSERT((gang_dora_count_ < 4u));
  KANACHAN_ASSERT((seat_ < 4u));
  KANACHAN_ASSERT((lizhi_list_[seat_] == 0u));
  KANACHAN_ASSERT((lizhi_delayed_ == 0u));
  KANACHAN_ASSERT((!angang_dora_delayed_));
  KANACHAN_ASSERT((!minggang_dora_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!first_zimo_[i]));
  }
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!yifa_[i]));
  }
  KANACHAN_ASSERT((!lingshang_kaihua_delayed_));
  KANACHAN_ASSERT((!qianggang_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!rong_delayed_[i]));
  }

  auto const [dapai_seat, dapai] = getLastDapai_();
  std::uint_fast8_t const relseat = (dapai_seat + 4u - seat_) % 4u - 1u;

  Kanachan::Shoupai &shoupai = shoupai_list_[seat_];
  shoupai.onDaminggang(relseat, dapai);

  progression_.append(1437 + seat_ * 111 + relseat * 37 + dapai);

  shoupai.onPostGang(/*in_lizhi = */false);

  minggang_dora_delayed_ = true;
  lingshang_kaihua_delayed_ = true;
}

std::uint_fast16_t RoundState::onAngang(
  std::uint_fast8_t const zimo_tile, std::uint_fast8_t const encode)
{
  KANACHAN_ASSERT((zimo_tile < 37u));
  KANACHAN_ASSERT((encode < 34u));

  KANACHAN_ASSERT((getNumLeftTiles() >= 1u));
  KANACHAN_ASSERT((lingshang_zimo_count_ < 4u));
  KANACHAN_ASSERT((gang_dora_count_ < 4u));
  KANACHAN_ASSERT((seat_ < 4u));
  KANACHAN_ASSERT((!lizhi_delayed_));
  KANACHAN_ASSERT((!angang_dora_delayed_));
  KANACHAN_ASSERT((!lingshang_kaihua_delayed_));
  KANACHAN_ASSERT((!qianggang_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!rong_delayed_[i]));
  }

  if (minggang_dora_delayed_) {
    KANACHAN_ASSERT((lingshang_zimo_count_ == gang_dora_count_ + 1u));
    ++gang_dora_count_;
  }
  minggang_dora_delayed_ = false;

  shoupai_list_[seat_].onAngang(zimo_tile, encode);

  progression_.append(1881 + seat_ * 34 + encode);

  qianggang_delayed_ = true;
  bool qianggang = false;
  for (std::uint_fast8_t i = seat_ + 1u; i <= seat_ + 3u; ++i) {
    std::uint_fast8_t const relseat = (seat_ + 4u - i) % 4u - 1u;

    python::list features = constructFeatures_(
      i % 4u, /*zimo_tile = */std::numeric_limits<std::uint_fast8_t>::max());
    features.append(progression_);

    Kanachan::Shoupai const &shoupai = shoupai_list_[i % 4u];
    python::list candidates = shoupai.getCandidatesOnAngang(relseat, encode);
    if (python::len(candidates) == 0u) {
      continue;
    }
    features.append(candidates);

    std::uint_fast16_t const action = selectAction_(i % 4u, features);
    if (543u <= action && action <= 545u) {
      // Qiang Gang (槍槓)
      KANACHAN_ASSERT((action - 543u == relseat));
      KANACHAN_ASSERT(((i + relseat + 1u) % 4u == seat_));
      rong_delayed_[i % 4] = true;
      qianggang = true;
      continue;
    }
    if (action == 221u) {
      // Skip
      continue;
    }
    KANACHAN_THROW<std::logic_error>(_1)
      << action << ": An invalid action on an gang.";
  }

  if (qianggang) {
    // 槍槓
    seat_ = std::numeric_limits<std::uint_fast8_t>::max();
    return 543u;
  }

  // 槍槓されなかったこの時点で槓が成立する．これにより一発が消失する．また，
  // これ以降の自摸が第一自摸と見なされなくなる．
  first_zimo_ = { false, false, false, false };
  yifa_ = { false, false, false, false };

  angang_dora_delayed_ = true;
  lingshang_kaihua_delayed_ = true;
  qianggang_delayed_ = false;

  shoupai_list_[seat_].onPostGang(lizhi_list_[seat_] >= 1u);

  return std::numeric_limits<std::uint_fast16_t>::max();
}

std::uint_fast16_t RoundState::onJiagang(
  std::uint_fast8_t const zimo_tile, std::uint_fast8_t const encode)
{
  KANACHAN_ASSERT((zimo_tile < 37u));
  KANACHAN_ASSERT((encode < 37u));

  KANACHAN_ASSERT((getNumLeftTiles() >= 1u));
  KANACHAN_ASSERT((lingshang_zimo_count_ < 4u));
  KANACHAN_ASSERT((gang_dora_count_ < 4u));
  KANACHAN_ASSERT((seat_ < 4u));
  KANACHAN_ASSERT((lizhi_list_[seat_] == 0u));
  KANACHAN_ASSERT((lizhi_delayed_ == 0u));
  KANACHAN_ASSERT((!angang_dora_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!first_zimo_[i]));
  }
  KANACHAN_ASSERT((!lingshang_kaihua_delayed_));
  KANACHAN_ASSERT((!qianggang_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!rong_delayed_[i]));
  }

  if (minggang_dora_delayed_) {
    KANACHAN_ASSERT((lingshang_zimo_count_ == gang_dora_count_ + 1u));
    ++gang_dora_count_;
  }
  minggang_dora_delayed_ = false;

  shoupai_list_[seat_].onJiagang(zimo_tile, encode);

  progression_.append(2017 + seat_ * 37 + encode);

  qianggang_delayed_ = true;
  bool qianggang = false;
  for (std::uint_fast8_t i = seat_ + 1u; i <= seat_ + 3u; ++i) {
    std::uint_fast8_t const relseat = (seat_ + 4u - i) % 4u - 1u;

    python::list features = constructFeatures_(
      i % 4u, /*zimo_tile = */std::numeric_limits<std::uint_fast8_t>::max());
    features.append(progression_);

    Kanachan::Shoupai const &shoupai = shoupai_list_[i % 4u];
    long const tool_config = encodeToolConfig_(i % 4u, /*rong = */true);
    python::list candidates
      = shoupai.getCandidatesOnJiagang(relseat, encode, tool_config);
    if (python::len(candidates) == 0u) {
      continue;
    }
    features.append(candidates);

    std::uint_fast16_t const action = selectAction_(i % 4u, features);

    if (543u <= action && action <= 545u) {
      // Qiang Gang (槍槓)
      KANACHAN_ASSERT((action - 543u == relseat));
      KANACHAN_ASSERT(((i + relseat + 1u) % 4u == seat_));
      rong_delayed_[i % 4] = true;
      qianggang = true;
      continue;
    }

    if (action == 221u) {
      // Skip
      continue;
    }

    KANACHAN_THROW<std::runtime_error>(_1)
      << action << ": An invalid action on jia gang.";
  }

  if (qianggang) {
    // 槍槓
    seat_ = std::numeric_limits<std::uint_fast8_t>::max();
    return 543u;
  }

  // 槍槓されなかったこの時点で槓が成立する．これにより一発が消失する．また，
  // これ以降の自摸が第一自摸と見なされなくなる．
  first_zimo_ = { false, false, false, false };
  yifa_ = { false, false, false, false };

  minggang_dora_delayed_ = true;
  lingshang_kaihua_delayed_ = true;
  qianggang_delayed_ = false;

  shoupai_list_[seat_].onPostGang(/*in_lizhi = */false);

  return std::numeric_limits<std::uint_fast16_t>::max();
}

bool RoundState::onHule(std::uint_fast8_t const zimo_tile, python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));
  KANACHAN_ASSERT((lingshang_zimo_count_ <= 4u));
  KANACHAN_ASSERT((gang_dora_count_ <= 4u));

  if (!result.attr("__contains__")("rounds")) {
    result["rounds"] = python::list();
  }
  python::list round_result;
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    round_result.append(python::dict());
  }
  python::extract<python::list>(result["rounds"])().append(round_result);

  if (zimo_tile != std::numeric_limits<std::uint_fast8_t>::max()) {
    // Zi Mo Hu (自摸和)
    KANACHAN_ASSERT((zimo_tile < 37u));
    KANACHAN_ASSERT((seat_ < 4u));
    KANACHAN_ASSERT((lizhi_delayed_ == 0u));
    KANACHAN_ASSERT((!angang_dora_delayed_));
    KANACHAN_ASSERT((!qianggang_delayed_));
    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      KANACHAN_ASSERT((!rong_delayed_[i]));
    }

    long const tool_config = encodeToolConfig_(seat_, /*rong = */false);
    auto const [fan, fu] = calculateHand_(seat_, zimo_tile, tool_config);

    if (seat_ == getJu()) {
      // 親の自摸和
      auto f = [this, round_result](std::int_fast32_t const delta_sanjia){
        {
          round_result[seat_]["result"] = 0; // 親の自摸和
          round_result[seat_]["lizhi"] = lizhi_list_[seat_] >= 1u ? 1 : 0;
          round_result[seat_]["menqian"] = isPlayerMenqian(seat_) ? 1 : 0;
          std::int_fast32_t const score
            = delta_sanjia * 3 + 300 * getBenChang() + 1000 * getNumLizhiDeposits();
          round_result[seat_]["delta_score"] = getPlayerDeltaScore(seat_) + score;
          game_state_.addPlayerScore(seat_, score);
        }

        for (std::uint_fast8_t i = 1u; i < 4u; ++i) {
          std::uint_fast8_t const seat = (seat_ + i) % 4u;
          round_result[seat]["result"] = 2; // 子=>親の被自摸和
          round_result[seat]["lizhi"] = lizhi_list_[seat] >= 1u ? 1 : 0;
          round_result[seat]["menqian"] = isPlayerMenqian(seat) ? 1 : 0;
          std::int_fast32_t const score = -delta_sanjia - 100 * getBenChang();
          round_result[seat]["delta_score"] = getPlayerDeltaScore(seat) + score;
          game_state_.addPlayerScore(seat, score);
        }
      };

      if (fan == 1u) {
        if (fu == 30u) {
          f(500);
        }
        else if (fu == 40u) {
          f(700);
        }
        else if (fu == 50u) {
          f(800);
        }
        else if (fu == 60u) {
          f(1000);
        }
        else if (fu == 70u) {
          f(1200);
        }
        else if (fu == 80u) {
          f(1300);
        }
        else if (fu == 90u) {
          f(1500);
        }
        else if (fu == 100u) {
          f(1600);
        }
        else {
          KANACHAN_ASSERT((fu == 110u));
          f(1800);
        }
      }
      else if (fan == 2u) {
        if (fu == 20u) {
          f(700);
        }
        else if (fu == 30u) {
          f(1000);
        }
        else if (fu == 40u) {
          f(1300);
        }
        else if (fu == 50u) {
          f(1600);
        }
        else if (fu == 60u) {
          f(2000);
        }
        else if (fu == 70u) {
          f(2300);
        }
        else if (fu == 80u) {
          f(2600);
        }
        else if (fu == 90u) {
          f(2900);
        }
        else if (fu == 100u) {
          f(3200);
        }
        else {
          KANACHAN_ASSERT((fu == 110u));
          f(3600);
        }
      }
      else if (fan == 3u) {
        if (fu == 20u) {
          f(1300);
        }
        else if (fu == 25u) {
          f(1600);
        }
        else if (fu == 30u) {
          f(2000);
        }
        else if (fu == 40u) {
          f(2600);
        }
        else if (fu == 50u) {
          f(3200);
        }
        else if (fu == 60u) {
          f(3900);
        }
        else {
          KANACHAN_ASSERT((fu >= 70u));
          f(4000);
        }
      }
      else if (fan == 4u) {
        if (fu == 20u) {
          f(2600);
        }
        else if (fu == 25u) {
          f(3200);
        }
        else if (fu == 30u) {
          f(3900);
        }
        else {
          KANACHAN_ASSERT((fu >= 40u));
          f(4000);
        }
      }
      else if (fan == 5u) {
        f(4000);
      }
      else if (6u <= fan && fan <= 7u) {
        f(6000);
      }
      else if (8u <= fan && fan <= 10u) {
        f(8000);
      }
      else if (11u <= fan && fan <= 12u) {
        f(12000);
      }
      else if (fan == 13u) {
        f(16000);
      }
      else if (fan == 26u) {
        f(32000);
      }
      else if (fan == 39u) {
        f(48000);
      }
      else if (fan == 52u) {
        f(64000);
      }
      else if (fan == 65u) {
        f(80000);
      }
      else if (fan == 78u) {
        f(96000);
      }
      else {
        KANACHAN_THROW<std::runtime_error>(_1)
          << '(' << static_cast<unsigned>(fan) << ", "
          << static_cast<unsigned>(fu) << ')';
      }

      {
        auto const [from, to] = checkDaSanyuanPao_();
        if (from != static_cast<std::uint_fast8_t>(-1) && to == seat_) {
          // 大三元の包が発生している．
          KANACHAN_ASSERT((from < 4u));

          for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
            // 包がある場合，本場は包の対象者が全額を払う．
            if (i == from) {
              std::int_fast32_t const score = -32000 - 200 * getBenChang();
              round_result[i]["delta_score"] += score;
              game_state_.addPlayerScore(i, score);
            }
            else if (i != seat_) {
              std::int_fast32_t const score = 16000 + 100 * getBenChang();
              round_result[i]["delta_score"] += score;
              game_state_.addPlayerScore(i, score);
            }
          }
        }
      }

      {
        auto const [from, to] = checkDaSixiPao_();
        if (from != static_cast<std::uint_fast8_t>(-1) && to == seat_) {
          // 大四喜の包が発生している．
          KANACHAN_ASSERT((from < 4u));

          for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
            // 包がある場合，本場は包の対象者が全額を払う．
            if (i == from) {
              std::int_fast32_t const score = -64000 - 200 * getBenChang();
              round_result[i]["delta_score"] += score;
              game_state_.addPlayerScore(i, score);
            }
            else if (i != seat_) {
              std::int_fast32_t const score = 32000 + 100 * getBenChang();
              round_result[i]["delta_score"] += score;
              game_state_.addPlayerScore(i, score);
            }
          }
        }
      }
    }
    else {
      // 子の自摸和
      auto f = [this, round_result](std::int_fast32_t const delta_zhuangjia, std::int_fast32_t const delta_sanjia){
        {
          round_result[seat_]["result"] = 1; // 子の自摸和
          round_result[seat_]["lizhi"] = lizhi_list_[seat_] >= 1u ? 1 : 0;
          round_result[seat_]["menqian"] = isPlayerMenqian(seat_) ? 1 : 0;
          std::int_fast32_t const score
            = delta_zhuangjia + delta_sanjia * 2 + 300 * getBenChang() + 1000 * getNumLizhiDeposits();
          round_result[seat_]["delta_score"] = getPlayerDeltaScore(seat_) + score;
          game_state_.addPlayerScore(seat_, score);
        }

        for (std::uint_fast8_t i = 1u; i < 4u; ++i) {
          std::uint_fast8_t const seat = (seat_ + i) % 4u;
          round_result[seat]["lizhi"] = lizhi_list_[seat] >= 1u ? 1 : 0;
          round_result[seat]["menqian"] = isPlayerMenqian(seat) ? 1 : 0;
          if (seat == getJu()) {
            round_result[seat]["result"] = 3; // 親=>子の被自摸和
            std::int_fast32_t const score = -delta_zhuangjia - 100 * getBenChang();
            round_result[seat]["delta_score"] = getPlayerDeltaScore(seat) + score;
            game_state_.addPlayerScore(seat, score);
          }
          else {
            round_result[seat]["result"] = 4; // 子=>子の被自摸和
            std::int_fast32_t const score = -delta_sanjia - 100 * getBenChang();
            round_result[seat]["delta_score"] = getPlayerDeltaScore(seat) + score;
            game_state_.addPlayerScore(seat, score);
          }
        }
      };

      if (fan == 1u) {
        if (fu == 30u) {
          f(500, 300);
        }
        else if (fu == 40u) {
          f(700, 400);
        }
        else if (fu == 50u) {
          f(800, 400);
        }
        else if (fu == 60u) {
          f(1000, 500);
        }
        else if (fu == 70u) {
          f(1200, 600);
        }
        else if (fu == 80u) {
          f(1300, 700);
        }
        else if (fu == 90u) {
          f(1500, 800);
        }
        else if (fu == 100u) {
          f(1600, 800);
        }
        else {
          KANACHAN_ASSERT((fu == 110u));
          f(1800, 900);
        }
      }
      else if (fan == 2u) {
        if (fu == 20u) {
          f(700, 400);
        }
        else if (fu == 30u) {
          f(1000, 500);
        }
        else if (fu == 40u) {
          f(1300, 700);
        }
        else if (fu == 50u) {
          f(1600, 800);
        }
        else if (fu == 60u) {
          f(2000, 1000);
        }
        else if (fu == 70u) {
          f(2300, 1200);
        }
        else if (fu == 80u) {
          f(2600, 1300);
        }
        else if (fu == 90u) {
          f(2900, 1500);
        }
        else if (fu == 100u) {
          f(3200, 1600);
        }
        else {
          KANACHAN_ASSERT((fu == 110u));
          f(3600, 1800);
        }
      }
      else if (fan == 3u) {
        if (fu == 20u) {
          f(1300, 700);
        }
        else if (fu == 25u) {
          f(1600, 800);
        }
        else if (fu == 30u) {
          f(2000, 1000);
        }
        else if (fu == 40u) {
          f(2600, 1300);
        }
        else if (fu == 50u) {
          f(3200, 1600);
        }
        else if (fu == 60u) {
          f(3900, 2000);
        }
        else {
          KANACHAN_ASSERT((fu >= 70u));
          f(4000, 2000);
        }
      }
      else if (fan == 4u) {
        if (fu == 20u) {
          f(2600, 1300);
        }
        else if (fu == 25u) {
          f(3200, 1600);
        }
        else if (fu == 30u) {
          f(3900, 2000);
        }
        else {
          KANACHAN_ASSERT((fu >= 40u));
          f(4000, 2000);
        }
      }
      else if (fan == 5u) {
        f(4000, 2000);
      }
      else if (6u <= fan && fan <= 7u) {
        f(6000, 3000);
      }
      else if (8u <= fan && fan <= 10u) {
        f(8000, 4000);
      }
      else if (11u <= fan && fan <= 12u) {
        f(12000, 6000);
      }
      else if (fan == 13u) {
        f(16000, 8000);
      }
      else if (fan == 26u) {
        f(32000, 16000);
      }
      else if (fan == 39u) {
        f(48000, 24000);
      }
      else if (fan == 52u) {
        f(64000, 32000);
      }
      else if (fan == 65u) {
        f(80000, 40000);
      }
      else if (fan == 78u) {
        f(96000, 48000);
      }
      else {
        KANACHAN_THROW<std::runtime_error>(_1)
          << '(' << static_cast<unsigned>(fan) << ", "
          << static_cast<unsigned>(fu) << ')';
      }

      {
        auto const [from, to] = checkDaSanyuanPao_();
        if (from != static_cast<std::uint_fast8_t>(-1) && to == seat_) {
          // 大三元の包が発生している．
          KANACHAN_ASSERT((from < 4u));

          for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
            // 包がある場合，本場は包の対象者が全額を払う．
            if (i == from) {
              if (i == getJu()) {
                // 包の対象者が親の場合．
                std::int_fast32_t const score = -16000 - 200 * getBenChang();
                round_result[i]["delta_score"] += score;
                game_state_.addPlayerScore(i, score);
              }
              else {
                // 包の対象者が子の場合．
                std::int_fast32_t const score = -24000 - 200 * getBenChang();
                round_result[i]["delta_score"] += score;
                game_state_.addPlayerScore(i, score);
              }
            }
            else if (i != seat_) {
              if (i == getJu()) {
                std::int_fast32_t const score = 16000 + 100 * getBenChang();
                round_result[i]["delta_score"] += score;
                game_state_.addPlayerScore(i, score);
              }
              else {
                std::int_fast32_t const score = 8000 + 100 * getBenChang();
                round_result[i]["delta_score"] += score;
                game_state_.addPlayerScore(i, score);
              }
            }
          }
        }
      }

      {
        auto const [from, to] = checkDaSixiPao_();
        if (from != static_cast<std::uint_fast8_t>(-1) && to == seat_) {
          // 大四喜の包が発生している．
          KANACHAN_ASSERT((from < 4u));

          for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
            // 包がある場合，本場は包の対象者が全額を払う．
            if (i == from) {
              if (i == getJu()) {
                // 包の対象者が親の場合．
                std::int_fast32_t const score = -32000 - 200 * getBenChang();
                round_result[i]["delta_score"] += score;
                game_state_.addPlayerScore(i, score);
              }
              else {
                // 包の対象者が子の場合．
                std::int_fast32_t const score = -48000 - 200 * getBenChang();
                round_result[i]["delta_score"] += score;
                game_state_.addPlayerScore(i, score);
              }
            }
            else if (i != seat_) {
              if (i == getJu()) {
                std::int_fast32_t const score = 32000 + 100 * getBenChang();
                round_result[i]["delta_score"] += score;
                game_state_.addPlayerScore(i, score);
              }
              else {
                std::int_fast32_t const score = 16000 + 100 * getBenChang();
                round_result[i]["delta_score"] += score;
                game_state_.addPlayerScore(i, score);
              }
            }
          }
        }
      }
    }
  }
  else {
    // Rong (栄和)
    KANACHAN_ASSERT(
      (zimo_tile == std::numeric_limits<std::uint_fast8_t>::max()));
    KANACHAN_ASSERT((seat_ == std::numeric_limits<std::uint_fast8_t>::max()));
    KANACHAN_ASSERT((!lingshang_kaihua_delayed_));

    {
      std::uint_fast8_t num_rong = 0u;
      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        if (rong_delayed_[i]) {
          ++num_rong;
        }
      }
      KANACHAN_ASSERT((1u <= num_rong && num_rong <= 3u));
    }

    auto const [dapai_seat, dapai] = getLastDapai_();

    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      python::object o = round_result[i];
      python::extract<python::dict> e(o);
      KANACHAN_ASSERT((e.check()));
      e()["delta_score"] = 0;
    }

    bool flag = true; // 本場・供託の上家取りフラグ
    for (std::uint_fast8_t i = 0u; i < 3u; ++i) {
      std::uint_fast8_t const seat = (dapai_seat + i + 1u) % 4u;

      if (!rong_delayed_[seat]) {
        continue;
      }

      long const tool_config = encodeToolConfig_(seat, /*rong = */true);
      auto const [fan, fu] = calculateHand_(seat, dapai, tool_config);

      auto f = [this, dapai_seat, seat, round_result](std::int_fast32_t const delta, bool const flag_) {
        round_result[seat]["result"] = (seat == getJu() ? 5 : 6);
        round_result[seat]["lizhi"] = (lizhi_list_[seat] >= 1u ? 1 : 0);
        round_result[seat]["menqian"] = (isPlayerMenqian(seat) ? 1 : 0);
        if (flag_) {
          std::int_fast32_t const score
            = delta + 300 * getBenChang() + 1000 * getNumLizhiDeposits();
          round_result[seat]["delta_score"] += getPlayerDeltaScore(seat) + score;
          game_state_.addPlayerScore(seat, score);
        }
        else {
          std::int_fast32_t const score = delta;
          round_result[seat]["delta_score"] += getPlayerDeltaScore(seat) + score;
          game_state_.addPlayerScore(seat, score);
        }

        for (std::uint_fast8_t j = 0u; j < 4u; ++j) {
          if (rong_delayed_[j]) {
            continue;
          }
          if (j == dapai_seat) {
            if (!round_result[j].attr("__contains__")("result")) {
              round_result[j]["result"] = (seat == getJu() ? 7 : 8);
            }
            else if (round_result[j]["result"] == 8) {
              round_result[j]["result"] = (seat == getJu() ? 7 : 8);
            }
            round_result[j]["lizhi"] = (lizhi_list_[j] >= 1u ? 1 : 0);
            round_result[j]["menqian"] = (isPlayerMenqian(j) ? 1 : 0);
            if (flag_) {
              std::int_fast32_t const score = -delta - 300 * getBenChang();
              round_result[j]["delta_score"] += getPlayerDeltaScore(j) + score;
              game_state_.addPlayerScore(j, score);
            }
            else {
              std::int_fast32_t const score = -delta;
              round_result[j]["delta_score"] += score;
              game_state_.addPlayerScore(j, score);
            }
            continue;
          }
          round_result[j]["result"] = 9;
          round_result[j]["lizhi"] = (lizhi_list_[j] >= 1u ? 1 : 0);
          round_result[j]["menqian"] = (isPlayerMenqian(j) ? 1 : 0);
          if (flag_) {
            round_result[j]["delta_score"] += getPlayerDeltaScore(j);
          }
        }
      };

      if (seat == getJu()) {
        // 親の栄和
        if (fan == 1u) {
          if (fu == 30u) {
            f(1500, flag);
          }
          else if (fu == 40u) {
            f(2000, flag);
          }
          else if (fu == 50u) {
            f(2400, flag);
          }
          else if (fu == 60u) {
            f(2900, flag);
          }
          else if (fu == 70u) {
            f(3400, flag);
          }
          else if (fu == 80u) {
            f(3900, flag);
          }
          else if (fu == 90u) {
            f(4400, flag);
          }
          else if (fu == 100u) {
            f(4800, flag);
          }
          else {
            KANACHAN_ASSERT((fu == 110u));
            f(5300, flag);
          }
        }
        else if (fan == 2u) {
          if (fu == 25u) {
            f(2400, flag);
          }
          else if (fu == 30u) {
            f(2900, flag);
          }
          else if (fu == 40u) {
            f(3900, flag);
          }
          else if (fu == 50u) {
            f(4800, flag);
          }
          else if (fu == 60u) {
            f(5800, flag);
          }
          else if (fu == 70u) {
            f(6800, flag);
          }
          else if (fu == 80u) {
            f(7700, flag);
          }
          else if (fu == 90u) {
            f(8700, flag);
          }
          else if (fu == 100u) {
            f(9600, flag);
          }
          else {
            KANACHAN_ASSERT((fu == 110u));
            f(10600, flag);
          }
        }
        else if (fan == 3u) {
          if (fu == 25u) {
            f(4800, flag);
          }
          else if (fu == 30u) {
            f(5800, flag);
          }
          else if (fu == 40u) {
            f(7700, flag);
          }
          else if (fu == 50u) {
            f(9600, flag);
          }
          else if (fu == 60u) {
            f(11600, flag);
          }
          else {
            KANACHAN_ASSERT((fu >= 70u));
            f(12000, flag);
          }
        }
        else if (fan == 4u) {
          if (fu == 25u) {
            f(9600, flag);
          }
          else if (fu == 30u) {
            f(11600, flag);
          }
          else {
            KANACHAN_ASSERT((fu >= 40u));
            f(12000, flag);
          }
        }
        else if (fan == 5u) {
          f(12000, flag);
        }
        else if (6u <= fan && fan <= 7u) {
          f(18000, flag);
        }
        else if (8u <= fan && fan <= 10u) {
          f(24000, flag);
        }
        else if (11u <= fan && fan <= 12u) {
          f(36000, flag);
        }
        else if (fan == 13u) {
          f(48000, flag);
        }
        else if (fan == 26u) {
          f(96000, flag);
        }
        else if (fan == 39u) {
          f(144000, flag);
        }
        else if (fan == 52u) {
          f(192000, flag);
        }
        else if (fan == 65u) {
          f(240000, flag);
        }
        else if (fan == 78u) {
          f(288000, flag);
        }
        else {
          KANACHAN_THROW<std::runtime_error>(_1)
            << '(' << static_cast<unsigned>(fan) << ", "
            << static_cast<unsigned>(fu) << ')';
        }
      }
      else {
        // 子の栄和
        if (fan == 1u) {
          if (fu == 30u) {
            f(1000, flag);
          }
          else if (fu == 40u) {
            f(1300, flag);
          }
          else if (fu == 50u) {
            f(1600, flag);
          }
          else if (fu == 60u) {
            f(2000, flag);
          }
          else if (fu == 70u) {
            f(2300, flag);
          }
          else if (fu == 80u) {
            f(2600, flag);
          }
          else if (fu == 90u) {
            f(2900, flag);
          }
          else if (fu == 100u) {
            f(3200, flag);
          }
          else {
            KANACHAN_ASSERT((fu == 110u));
            f(3600, flag);
          }
        }
        else if (fan == 2u) {
          if (fu == 25u) {
            f(1600, flag);
          }
          else if (fu == 30u) {
            f(2000, flag);
          }
          else if (fu == 40u) {
            f(2600, flag);
          }
          else if (fu == 50u) {
            f(3200, flag);
          }
          else if (fu == 60u) {
            f(3900, flag);
          }
          else if (fu == 70u) {
            f(4500, flag);
          }
          else if (fu == 80u) {
            f(5200, flag);
          }
          else if (fu == 90u) {
            f(5800, flag);
          }
          else if (fu == 100u) {
            f(6400, flag);
          }
          else {
            KANACHAN_ASSERT((fu == 110u));
            f(7100, flag);
          }
        }
        else if (fan == 3u) {
          if (fu == 25u) {
            f(3200, flag);
          }
          else if (fu == 30u) {
            f(3900, flag);
          }
          else if (fu == 40u) {
            f(5200, flag);
          }
          else if (fu == 50u) {
            f(6400, flag);
          }
          else if (fu == 60u) {
            f(7700, flag);
          }
          else {
            KANACHAN_ASSERT((fu >= 70u));
            f(8000, flag);
          }
        }
        else if (fan == 4u) {
          if (fu == 25u) {
            f(6400, flag);
          }
          else if (fu == 30u) {
            f(7700, flag);
          }
          else {
            KANACHAN_ASSERT((fu >= 40u));
            f(8000, flag);
          }
        }
        else if (fan == 5u) {
          f(8000, flag);
        }
        else if (6u <= fan && fan <= 7u) {
          f(12000, flag);
        }
        else if (8u <= fan && fan <= 10u) {
          f(16000, flag);
        }
        else if (11u <= fan && fan <= 12u) {
          f(24000, flag);
        }
        else if (fan == 13u) {
          f(32000, flag);
        }
        else if (fan == 26u) {
          f(64000, flag);
        }
        else if (fan == 39u) {
          f(96000, flag);
        }
        else if (fan == 52u) {
          f(128000, flag);
        }
        else if (fan == 65u) {
          f(160000, flag);
        }
        else if (fan == 78u) {
          f(192000, flag);
        }
        else {
          KANACHAN_THROW<std::runtime_error>(_1)
            << '(' << static_cast<unsigned>(fan) << ", "
            << static_cast<unsigned>(fu) << ')';
        }
      }

      flag = false;
    }

    for (;;) {
      auto const [from, to] = checkDaSanyuanPao_();
      if (from == static_cast<std::uint_fast8_t>(-1)) {
        break;
      }
      KANACHAN_ASSERT((from < 4u));
      KANACHAN_ASSERT((to < 4u));
      if (!rong_delayed_[to]) {
        break;
      }

      // 大三元の包が発生している．
      std::int_fast32_t const score_base
        = (to == game_state_.getJu() ? 24000 : 16000);
      // 包を伴う和了に本場分の点数が付帯しているかどうかのフラグ．
      bool const flag = [&]() -> bool {
        for (std::uint_fast8_t i = dapai_seat + 1u; i < dapai_seat + 4u; ++i) {
          std::uint_fast8_t const seat = i % 4u;
          if (rong_delayed_[seat]) {
            return seat == to;
          }
        }
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }();
      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        // 包を伴う和了に本場分の点数が付帯する場合，包の対象者が本場分の点数の
        // 全額を支払う．ただし， `flag == false` の場合，
        // ダブロン・トリプルロンによる上家取りにより包を伴う和了に本場分の
        // 点数が付帯していないことに注意する．
        if (i == from) {
          std::int_fast32_t const score
            = -score_base + (flag ? -300 * getBenChang() : 0);
          round_result[i]["delta_score"] += score;
          game_state_.addPlayerScore(i, score);
        }
        if (i == dapai_seat){
          std::int_fast32_t const score
            = score_base + (flag ? 300 * getBenChang() : 0);
          round_result[i]["delta_score"] += score;
          game_state_.addPlayerScore(i, score);
        }
      }
      break;
    }

    for (;;) {
      auto const [from, to] = checkDaSixiPao_();
      if (from == static_cast<std::uint_fast8_t>(-1)) {
        break;
      }
      KANACHAN_ASSERT((from < 4u));
      KANACHAN_ASSERT((to < 4u));
      if (!rong_delayed_[to]) {
        break;
      }

      // 大四喜の包が発生している．
      std::int_fast32_t const score_base
        = (to == game_state_.getJu() ? 48000 : 32000);
      // 包を伴う和了に本場分の点数が付帯しているかどうかのフラグ．
      bool const flag = [&]() -> bool {
        for (std::uint_fast8_t i = dapai_seat + 1u; i < dapai_seat + 4u; ++i) {
          std::uint_fast8_t const seat = i % 4u;
          if (rong_delayed_[seat]) {
            return seat == to;
          }
        }
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }();
      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        // 包を伴う和了に本場分の点数が付帯する場合，包の対象者が本場分の点数の
        // 全額を支払う．ただし， `flag == false` の場合，
        // ダブロン・トリプルロンによる上家取りにより包を伴う和了に本場分の
        // 点数が付帯していないことに注意する．
        if (i == from) {
          std::int_fast32_t const score
            = -score_base + (flag ? -300 * getBenChang() : 0);
          round_result[i]["delta_score"] += score;
          game_state_.addPlayerScore(i, score);
        }
        if (i == dapai_seat){
          std::int_fast32_t const score
            = score_base + (flag ? 300 * getBenChang() : 0);
          round_result[i]["delta_score"] += score;
          game_state_.addPlayerScore(i, score);
        }
      }
      break;
    }
  }

  // 飛び終了
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    if (game_state_.getPlayerScore(i) < 0) {
      return true;
    }
  }

  std::uint_fast8_t const chang = game_state_.getChang();
  std::uint_fast8_t const ju = game_state_.getJu();

  if (game_state_.isDongfengZhan()) {
    // 東風戦
    if (chang == 1u && ju == 3u) {
      // 南4局
      bool suddendeath = false;
      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        if (getPlayerScore(i) >= 30000) {
          suddendeath = true;
          break;
        }
      }
      if (suddendeath) {
        // いずれかのプレーヤの点数が30000点以上である．
        if (seat_ != ju && !rong_delayed_[ju]) {
          // 親が和了していない．
          return true;
        }
        if (getPlayerRanking(ju) == 0u) {
          // 親が1位である．
          return true;
        }
        // いずれかのプレーヤの点数が30000点以上で，親が和了しているが，
        // 親が1位ではない．
        game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
        return false;
      }
      // どのプレーヤも30000点未満である．
      if (seat_ == ju || rong_delayed_[ju]) {
        // 親が和了している．
        game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
        return false;
      }
      return true;
    }
    if (chang == 0u && ju == 3u || chang == 1u) {
      // 東4局または南入
      bool suddendeath = false;
      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        if (getPlayerScore(i) >= 30000) {
          suddendeath = true;
          break;
        }
      }
      if (suddendeath) {
        // いずれかのプレーヤの点数が30000点以上である．
        if (seat_ != ju && !rong_delayed_[ju]) {
          // 親が和了していない．
          return true;
        }
        if (getPlayerRanking(ju) == 0u) {
          // 親が1位である．
          return true;
        }
        // いずれかのプレーヤの点数が30000点以上で，親が和了しているが，
        // 親が1位ではない．
        game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
        return false;
      }
      // どのプレーヤも30000点未満である．
      if (seat_ == ju || rong_delayed_[ju]) {
        game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
      }
      else {
        game_state_.onLunzhuang(GameState::RoundEndStatus::hule);
      }
      return false;
    }
    if (seat_ == getJu()) {
      // 親の自摸和による連荘．
      game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
    }
    else if (seat_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      // 子の自摸和による輪荘．
      game_state_.onLunzhuang(GameState::RoundEndStatus::hule);
    }
    else if (rong_delayed_[ju]) {
      // 親の栄和（ダブロン・トリプルロンを含む）による連荘．
      game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
    }
    else {
      // 子の栄和による輪荘．親が栄和していない場合に限る．
      game_state_.onLunzhuang(GameState::RoundEndStatus::hule);
    }
    return false;
  }

  KANACHAN_ASSERT((!game_state_.isDongfengZhan()));
  if (chang == 2u && ju == 3u) {
    // 西4局
    bool suddendeath = false;
    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      if (getPlayerScore(i) >= 30000) {
        suddendeath = true;
        break;
      }
    }
    if (suddendeath) {
      // いずれかのプレーヤの点数が30000点以上である．
      if (seat_ != ju && !rong_delayed_[ju]) {
        // 親が和了していない．
        return true;
      }
      if (getPlayerRanking(ju) == 0u) {
        // 親が1位である．
        return true;
      }
      // いずれかのプレーヤの点数が30000点以上で，親が和了しているが，
      // 親が1位ではない．
      game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
      return false;
    }
    // どのプレーヤも30000点未満である．
    if (seat_ == ju || rong_delayed_[ju]) {
      // 親が和了している．
      game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
      return false;
    }
    return true;
  }
  if (chang == 1u && ju == 3u || chang == 2u) {
    // 南4局または西入
    bool suddendeath = false;
    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      if (getPlayerScore(i) >= 30000) {
        suddendeath = true;
        break;
      }
    }
    if (suddendeath) {
      // いずれかのプレーヤの点数が30000点以上である．
      if (seat_ != ju && !rong_delayed_[ju]) {
        // ラス親が和了していない．
        return true;
      }
      if (getPlayerRanking(ju) == 0u) {
        // ラス親が1位である．
        return true;
      }
      // いずれかのプレーヤの点数が30000点以上で，ラス親が和了しているが，
      // ラス親が1位ではない．
      game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
      return false;
    }
    // どのプレーヤも30000点未満である．
    if (seat_ == ju || rong_delayed_[ju]) {
      game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
    }
    else {
      game_state_.onLunzhuang(GameState::RoundEndStatus::hule);
    }
    return false;
  }
  if (zimo_tile != std::numeric_limits<std::uint_fast8_t>::max()) {
    if (seat_ == ju) {
      // 親の自摸和による連荘．
      game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
    }
    else {
      // 子の自摸和による輪荘．
      game_state_.onLunzhuang(GameState::RoundEndStatus::hule);
    }
  }
  else if (rong_delayed_[ju]) {
    // 親の栄和（ダブロン・トリプルロンを含む）による連荘．
    game_state_.onLianzhuang(GameState::RoundEndStatus::hule);
  }
  else {
    // 子の栄和による輪荘．親が栄和していない場合に限る．
    game_state_.onLunzhuang(GameState::RoundEndStatus::hule);
  }
  return false;
}

bool RoundState::onHuangpaiPingju(python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));

  KANACHAN_ASSERT((getNumLeftTiles() == 0u));
  KANACHAN_ASSERT((lingshang_zimo_count_ <= 4u));
  KANACHAN_ASSERT((gang_dora_count_ <= 4u));
  KANACHAN_ASSERT((lingshang_zimo_count_ == gang_dora_count_));
  KANACHAN_ASSERT((seat_ < 4u));
  KANACHAN_ASSERT((lizhi_delayed_ == 0u));
  KANACHAN_ASSERT((!angang_dora_delayed_));
  KANACHAN_ASSERT((!minggang_dora_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!first_zimo_[i]));
  }
  KANACHAN_ASSERT((!lingshang_kaihua_delayed_));
  KANACHAN_ASSERT((!qianggang_delayed_));
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    KANACHAN_ASSERT((!rong_delayed_[i]));
  }

  if (!result.attr("__contains__")("rounds")) {
    result["rounds"] = python::list();
  }
  python::list round_result;
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    round_result.append(python::dict());
    round_result[-1]["delta_score"] = 0;
  }
  python::extract<python::list>(result["rounds"])().append(round_result);

  bool liuju_manguan = false;
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    if (isPlayerTingpai(i)) {
      round_result[i]["result"] = 11;
    }
    else {
      round_result[i]["result"] = 10;
    }
    round_result[i]["lizhi"] = (lizhi_list_[i] >= 1u ? 1 : 0);
    round_result[i]["menqian"] = isPlayerMenqian(i) ? 1 : 0;
    if (checkPlayerLiujuManguan(i)) {
      if (i == getJu()) {
        for (std::uint_fast8_t j = 0u; j < 4u; ++j) {
          if (j == i) {
            round_result[j]["delta_score"]
              += (liuju_manguan ? 0 : getPlayerDeltaScore(j)) + 12000;
            game_state_.addPlayerScore(j, 12000);
          }
          else {
            round_result[j]["delta_score"]
              += (liuju_manguan ? 0 : getPlayerDeltaScore(j)) - 4000;
            game_state_.addPlayerScore(j, -4000);
          }
        }
      }
      else {
        for (std::uint_fast8_t j = 0u; j < 4u; ++j) {
          if (j == i) {
            round_result[j]["delta_score"]
              += (liuju_manguan ? 0 : getPlayerDeltaScore(j)) + 8000;
            game_state_.addPlayerScore(j, 8000);
          }
          else if (j == getJu()) {
            round_result[j]["delta_score"]
              += (liuju_manguan ? 0 : getPlayerDeltaScore(j)) - 4000;
            game_state_.addPlayerScore(j, -4000);
          }
          else {
            round_result[j]["delta_score"]
              += (liuju_manguan ? 0 : getPlayerDeltaScore(j)) - 2000;
            game_state_.addPlayerScore(j, -2000);
          }
        }
      }
      liuju_manguan = true;
    }
  }

  if (!liuju_manguan) {
    auto const [delta_buting, delta_tingpai] = [this]() -> std::pair<std::int_fast32_t, std::int_fast32_t> {
      std::uint_fast8_t count = 0u;
      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        if (isPlayerTingpai(i)) {
          ++count;
        }
      }
      using Result = std::pair<std::int_fast32_t, std::int_fast32_t>;
      switch (count) {
      case 0u:
        return Result(0, 0);
      case 1u:
        return Result(1000, 3000);
      case 2u:
        return Result(1500, 1500);
      case 3u:
        return Result(3000, 1000);
      case 4u:
        return Result(0, 0);
      default:
        KANACHAN_THROW<std::logic_error>(_1) << count;
      }
    }();

    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      if (isPlayerTingpai(i)) {
        round_result[i]["delta_score"] += getPlayerDeltaScore(i) + delta_tingpai;
        game_state_.addPlayerScore(i, delta_tingpai);
      }
      else {
        round_result[i]["delta_score"] += getPlayerDeltaScore(i) - delta_buting;
        game_state_.addPlayerScore(i, -delta_buting);
      }
    }
  }

  // 飛び終了
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    if (game_state_.getPlayerScore(i) < 0) {
      settleLizhiDeposits_();
      return true;
    }
  }

  std::uint_fast8_t const chang = game_state_.getChang();
  std::uint_fast8_t const ju = game_state_.getJu();

  if (game_state_.isDongfengZhan()) {
    // 東風戦
    if (chang == 1u && ju == 3u) {
      // 南4局
      bool suddendeath = false;
      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        if (getPlayerScore(i) >= 30000) {
          suddendeath = true;
          break;
        }
      }
      if (suddendeath) {
        // いずれかのプレーヤの点数が30000点以上である．
        if (!isPlayerTingpai(ju)) {
          // 親が聴牌していない．
          settleLizhiDeposits_();
          return true;
        }
        if (getPlayerRanking(ju) == 0u) {
          // 親が1位である．
          settleLizhiDeposits_();
          return true;
        }
        // いずれかのプレーヤの点数が30000点以上で，親が聴牌しているが，
        // 親が1位ではない．
        game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
        return false;
      }
      // どのプレーヤも30000点未満である．
      if (isPlayerTingpai(ju)) {
        // 親が聴牌している．
        game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
        return false;
      }
      // 親が不聴である．
      settleLizhiDeposits_();
      return true;
    }
    if (chang == 0u && ju == 3u || chang == 1u) {
      if (isPlayerTingpai(ju)) {
        if (getPlayerRanking(ju) == 0u && getPlayerScore(ju) >= 30000) {
          settleLizhiDeposits_();
          return true;
        }
        game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
        return false;
      }
      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        if (getPlayerScore(i) >= 30000) {
          settleLizhiDeposits_();
          return true;
        }
      }
      game_state_.onLunzhuang(GameState::RoundEndStatus::huangpai_pingju);
      return false;
    }
    if (isPlayerTingpai(ju)) {
      game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
    }
    else {
      game_state_.onLunzhuang(GameState::RoundEndStatus::huangpai_pingju);
    }
    return false;
  }

  // 半荘戦
  KANACHAN_ASSERT((!game_state_.isDongfengZhan()));
  if (chang == 2u && ju == 3u) {
    // 西4局
    bool suddendeath = false;
    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      if (getPlayerScore(i) >= 30000) {
        suddendeath = true;
        break;
      }
    }
    if (suddendeath) {
      // いずれかのプレーヤの点数が30000点以上である．
      if (!isPlayerTingpai(ju)) {
        // 親が聴牌していない．
        settleLizhiDeposits_();
        return true;
      }
      if (getPlayerRanking(ju) == 0u) {
        // 親が1位である．
        settleLizhiDeposits_();
        return true;
      }
      // いずれかのプレーヤの点数が30000点以上で，親が聴牌しているが，
      // 親が1位ではない．
      game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
      return false;
    }
    // どのプレーヤも30000点未満である．
    if (isPlayerTingpai(ju)) {
      // 親が聴牌している．
      game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
      return false;
    }
    // 親が不聴である．
    settleLizhiDeposits_();
    return true;
  }
  if (chang == 1u && ju == 3u || chang == 2u) {
    // 南4局または西入
    bool suddendeath = false;
    for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
      if (getPlayerScore(i) >= 30000) {
        suddendeath = true;
        break;
      }
    }
    if (suddendeath) {
      // いずれかのプレーヤの点数が30000点以上である．
      if (!isPlayerTingpai(ju)) {
        // 親が聴牌していない．
        settleLizhiDeposits_();
        return true;
      }
      if (getPlayerRanking(ju) == 0u) {
        // 親が1位である．
        settleLizhiDeposits_();
        return true;
      }
      // いずれかのプレーヤの点数が30000点以上で，親が聴牌しているが，
      // 親が1位ではない．
      game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
      return false;
    }
    // どのプレーヤも30000点未満である．
    if (isPlayerTingpai(ju)) {
      // 親が聴牌している．
      game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
    }
    else {
      // 親が不聴である．
      game_state_.onLunzhuang(GameState::RoundEndStatus::huangpai_pingju);
    }
    return false;
  }
  if (isPlayerTingpai(ju)) {
    // 親が聴牌している．
    game_state_.onLianzhuang(GameState::RoundEndStatus::huangpai_pingju);
  }
  else {
    // 親が不聴である．
    game_state_.onLunzhuang(GameState::RoundEndStatus::huangpai_pingju);
  }
  return false;
}

void RoundState::onLiuju(python::dict result)
{
  KANACHAN_ASSERT((!result.is_none()));

  if (lizhi_delayed_ != 0u) {
    // 直前の立直を成立させる．四風連打，四槓散了，四家立直の場合．
    lizhi_list_[seat_] = lizhi_delayed_;
    yifa_[seat_] = true;
    game_state_.onSuccessfulLizhi(seat_);
    lizhi_delayed_ = 0u;
  }

  if (!result.attr("__contains__")("rounds")) {
    result["rounds"] = python::list();
  }
  python::list round_result;
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    round_result.append(python::dict());
  }
  python::extract<python::list>(result["rounds"])().append(round_result);

  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    round_result[i]["result"] = 12;
    round_result[i]["lizhi"] = (lizhi_list_[i] >= 1u ? 1 : 0);
    round_result[i]["menqian"] = isPlayerMenqian(i) ? 1 : 0;
    round_result[i]["delta_score"] = getPlayerDeltaScore(i);
  }

  game_state_.onLianzhuang(GameState::RoundEndStatus::liuju);
}

} // namespace Kanachan
