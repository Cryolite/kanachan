#include "annotation/player_state.hpp"

#include "annotation/utility.hpp"
#include "common/throw.hpp"
#include "common/mahjongsoul.pb.h"
#include <iostream>
#include <algorithm>
#include <array>
#include <string_view>
#include <string>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>


namespace Kanachan{

namespace{

using std::placeholders::_1;

template<class Iterator>
std::uint_fast8_t getRank_(std::uint_fast8_t seat, Iterator first, Iterator last)
{
  if (seat >= 4u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "seat = " << static_cast<unsigned>(seat);
  }

  if (last - first != 4) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "size = " << last - first;
  }

  using Pair = std::pair<std::uint_fast8_t, std::uint_fast32_t>;
  std::array<Pair, 4u> data{
    Pair(0u, first[0u]), Pair(1u, first[1u]), Pair(2u, first[2u]), Pair(3u, first[3u])
  };
  std::sort(
    data.begin(), data.end(),
    [](Pair const &p, Pair const &q) -> bool
    {
      if (p.second < q.second) {
        return false;
      }
      if (p.second > q.second) {
        return true;
      }
      return p.first < q.first;
    });
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    if (data[i].first == seat) {
      return i;
    }
  }

  KANACHAN_THROW<std::logic_error>("");
}

constexpr std::array<std::uint_fast8_t, 38u> hand_offsets_{
    0u,   1u,   5u,   9u,  13u,  17u,  20u,  24u,  28u,  32u,
   36u,  37u,  41u,  45u,  49u,  53u,  56u,  60u,  64u,  68u,
   72u,  73u,  77u,  81u,  85u,  89u,  92u,  96u, 100u, 104u,
  108u, 112u, 116u, 120u, 124u, 128u, 132u, 136u
};

template<class Iterator>
std::array<std::uint_fast8_t, 136u> makeHand_(Iterator first, Iterator last)
{
  if (last - first != 13u) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "size = " << last - first;
  }
  std::array<std::uint_fast8_t, 37u> counts;
  for (auto &e : counts) e = 0u;
  std::array<std::uint_fast8_t, 136u> result;
  for (auto &e : result) e = 0u;
  for (; first != last; ++first) {
    std::uint_fast8_t pai = Kanachan::pai2Num(*first);
    if (pai >= 37u) {
      KANACHAN_THROW<std::logic_error>(_1) << "pai = " << static_cast<unsigned>(pai);
    }
    std::uint_fast8_t i = hand_offsets_[pai] + counts[pai];
    result[i] = 1u;
    ++counts[pai];
    if (counts[pai] > hand_offsets_[pai + 1u] - hand_offsets_[pai]) {
      KANACHAN_THROW<std::logic_error>(_1)
        << "# of " << static_cast<unsigned>(pai) << " = " << counts[pai];
    }
  }
  return result;
}

} // namespace *unnamed*

PlayerState::PlayerState(
  std::uint_fast8_t const seat,
  std::uint_fast32_t const mode_id,
  std::uint_fast32_t const level,
  std::uint_fast8_t const game_rank,
  std::int_fast32_t const game_score,
  std::int_fast32_t const delta_grading_point)
  : seat_(seat),
    game_rank_(game_rank),
    game_score_(game_score),
    delta_grading_point_(delta_grading_point)
{
  if (seat >= 4u) {
    KANACHAN_THROW<std::invalid_argument>(_1)
      << "seat = " << static_cast<unsigned>(seat);
  }

  switch (mode_id) {
  case 2u:
    // 段位戦・銅の間・四人東風戦
    room_ = 0u;
    num_rounds_type_ = 0;
    break;
  case 3u:
    // 段位戦・銅の間・四人半荘戦
    room_ = 0u;
    num_rounds_type_ = 1u;
    break;
  case 5u:
    // 段位戦・銀の間・四人東風戦
    room_ = 1u;
    num_rounds_type_ = 0u;
    break;
  case 6u:
    // 段位戦・銀の間・四人半荘戦
    room_ = 1u;
    num_rounds_type_ = 1u;
    break;
  case 8u:
    // 段位戦・金の間・四人東風戦
    room_ = 2u;
    num_rounds_type_ = 0u;
    break;
  case 9u:
    // 段位戦・金の間・四人半荘戦
    room_ = 2u;
    num_rounds_type_ = 1u;
    break;
  case 11u:
    // 段位戦・玉の間・四人東風戦
    room_ = 3u;
    num_rounds_type_ = 0u;
    break;
  case 12u:
    // 段位戦・玉の間・四人半荘戦
    room_ = 3u;
    num_rounds_type_ = 1u;
    break;
  case 15u:
    // 段位戦・王座の間・四人東風戦
    room_ = 4u;
    num_rounds_type_ = 0u;
    break;
  case 16u:
    // 段位戦・王座の間・四人半荘戦
    room_ = 4u;
    num_rounds_type_ = 1u;
    break;
  default:
    KANACHAN_THROW<std::invalid_argument>(_1) << "mode_id = " << mode_id;
  }

  switch (level) {
  case 10101u:
    // 四麻初心1
    level_ = 0u;
    break;
  case 10102u:
    // 四麻初心2
    level_ = 1u;
    break;
  case 10103u:
    // 四麻初心3
    level_ = 2u;
    break;
  case 10201u:
    // 四麻雀士1
    level_ = 3u;
    break;
  case 10202u:
    // 四麻雀士2
    level_ = 4u;
    break;
  case 10203u:
    // 四麻雀士3
    level_ = 5u;
    break;
  case 10301u:
    // 四麻雀傑1
    level_ = 6u;
    break;
  case 10302u:
    // 四麻雀傑2
    level_ = 7u;
    break;
  case 10303u:
    // 四麻雀傑3
    level_ = 8u;
    break;
  case 10401u:
    // 四麻雀豪1
    level_ = 9u;
    break;
  case 10402u:
    // 四麻雀豪2
    level_ = 10u;
    break;
  case 10403u:
    // 四麻雀豪3
    level_ = 11u;
    break;
  case 10501u:
    // 四麻雀聖1
    level_ = 12u;
    break;
  case 10502u:
    // 四麻雀聖2
    level_ = 13u;
    break;
  case 10503u:
    // 四麻雀聖3
    level_ = 14u;
    break;
  case 10601u:
    // 四麻魂天 (2021/08/26 魂珠導入以前)
    level_ = 15u;
    break;
  case 10701u:
  case 10702u:
  case 10703u:
  case 10704u:
  case 10705u:
  case 10706u:
  case 10707u:
  case 10708u:
  case 10709u:
  case 10710u:
  case 10711u:
  case 10712u:
  case 10713u:
  case 10714u:
  case 10715u:
  case 10716u:
  case 10717u:
  case 10718u:
  case 10719u:
  case 10720u:
    // 四麻魂天 (2021/08/26 魂珠導入以降)
    level_ = 15u;
    break;
  default:
    KANACHAN_THROW<std::invalid_argument>(_1) << "level = " << level;
  }
}

void PlayerState::onNewRound(lq::RecordNewRound const &record)
{
  chang_ = record.chang();
  if (chang_ >= 3u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << "A broken data: chang = " << static_cast<unsigned>(chang_);
  }

  ju_ = record.ju();
  if (ju_ >= 4u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << "A broken data: ju = " << static_cast<unsigned>(ju_);
  }

  ben_ = record.ben();
  liqibang_ = record.liqibang();

  if (record.doras().size() != 1u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << "A broken data: # of dora indicators = " << record.doras().size();
  }
  dora_indicators_[0u] = Kanachan::pai2Num(record.doras()[0u]);
  for (std::uint_fast8_t i = 1u; i <= 4u; ++i) {
    dora_indicators_[i] = std::numeric_limits<std::uint_fast8_t>::max();
  }

  if (record.left_tile_count() != 69u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << "A broken data: # of left tiles = " << record.left_tile_count();
  }
  count_ = record.left_tile_count();

  if (record.scores().size() != 4u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << "A broken data: # of scores = " << record.scores().size();
  }
  initial_score_ = record.scores()[seat_];
  score_ = initial_score_;

  auto const &tiles = [&]() {
    switch (seat_) {
    case 0u:
      return record.tiles0();
    case 1u:
      return record.tiles1();
    case 2u:
      return record.tiles2();
    case 3u:
      return record.tiles3();
    default:
      KANACHAN_THROW<std::logic_error>(_1) << "seat = " << static_cast<unsigned>(seat_);
    }
  }();
  if (tiles.size() == 14u) {
    hand_ = makeHand_(tiles.cbegin(), tiles.cend() - 1u);
    zimo_pai_ = Kanachan::pai2Num(tiles.cbegin()[13u]);
  }
  else {
    if (tiles.size() != 13u) {
      KANACHAN_THROW<std::runtime_error>(_1) << "# of tiles = " << tiles.size();
    }
    hand_ = makeHand_(tiles.cbegin(), tiles.cend());
    zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max();
  }
}

void PlayerState::onZimo(lq::RecordDealTile const &record)
{
  if (record.has_liqi()) {
    auto const &liqi = record.liqi();
    if (liqi.seat() != (3u + record.seat()) % 4u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << "A broken data: TO-BE = " << (3u + record.seat()) % 4u << ", AS-IS = " << liqi.seat();
    }
    if (liqi.failed()) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    if (seat_ == liqi.seat()) {
      score_ -= 1000;
      if (score_ != liqi.score()) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << "A broken data: TO-BE = " << score_ << ", AS-IS = " << liqi.score();
      }
    }
    ++liqibang_;
  }

  if (record.doras().size() != 0u) {
    if (record.doras().size() > 5u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << "A broken data: # of dora indicators = " << record.doras().size();
    }
    for (std::uint_fast8_t i = 0u; i < record.doras().size(); ++i) {
      dora_indicators_[i] = Kanachan::pai2Num(record.doras()[i]);
    }
  }

  --count_;
  if (record.left_tile_count() != count_) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << "A broken data: TO-BE = " << static_cast<unsigned>(count_)
      << ", AS-IS = " << record.left_tile_count();
  }

  if (record.seat() != seat_) {
    return;
  }

  if (zimo_pai_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    KANACHAN_THROW<std::logic_error>(_1) << "zimo_pai_ = " << static_cast<unsigned>(zimo_pai_);
  }
  zimo_pai_ = Kanachan::pai2Num(record.tile());
}

void PlayerState::onDapai(lq::RecordDiscardTile const &record)
{
  if (record.doras().size() != 0u) {
    if (record.doras().size() > 5u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << "A broken data: # of dora indicators = " << record.doras().size();
    }
    for (std::uint_fast8_t i = 0u; i < record.doras().size(); ++i) {
      dora_indicators_[i] = Kanachan::pai2Num(record.doras()[i]);
    }
  }

  if (record.seat() != seat_) {
    return;
  }

  std::uint_fast8_t const dapai = Kanachan::pai2Num(record.tile());
  if (record.moqie()) {
    if (zimo_pai_ != dapai) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << "A broken data: TO-BE" << static_cast<unsigned>(zimo_pai_)
        << "AS-IS" << static_cast<unsigned>(dapai);
    }
    zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max();
  }
  else {
    {
      bool success = false;
      for (std::uint_fast8_t i = hand_offsets_[dapai + 1u]; i > hand_offsets_[dapai];) {
        --i;
        if (hand_[i] == 1u) {
          hand_[i] = 0u;
          success = true;
          break;
        }
        else if (hand_[i] != 0u) {
          KANACHAN_THROW<std::logic_error>("");
        }
      }
      if (!success) {
        if (count_ != 69u || zimo_pai_ != dapai) {
          KANACHAN_THROW<std::runtime_error>(_1)
            << "A broken data: count = " << static_cast<unsigned>(count_)
            << ", dapai = " << static_cast<unsigned>(dapai);
        }
        zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max();
      }
    }

    if (zimo_pai_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      handIn_();
    }
  }
}

void PlayerState::onChiPengGang(lq::RecordChiPengGang const &record)
{
  if (record.has_liqi()) {
    // 直前の立直がロンされずに立直が成立した．

    // 立直宣言者を探す．
    std::uint_fast8_t liqi_seat = std::numeric_limits<std::uint_fast8_t>::max();
    if (record.froms().size() == 3u) {
      // チー・ポンの場合．
      if (record.type() != 0u && record.type() != 1u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      // `record.froms()` の最後の要素が鳴いた牌を打牌した席でつまりは立直宣言者の席．
      liqi_seat = record.froms()[2u];
    }
    else if (record.froms().size() == 4u) {
      // 大明槓の場合．
      if (record.type() != 2u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      // `record.froms()` の最後の要素が鳴いた牌を打牌した席でつまりは立直宣言者の席．
      liqi_seat = record.froms()[3u];
    }

    // 立直宣言者の持ち点を1000点減らして供託本数を1本増やす．
    auto const &liqi = record.liqi();
    if (liqi_seat != liqi.seat()) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    if (seat_ == liqi_seat) {
      score_ -= 1000;
      if (score_ != liqi.score()) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
    }
    ++liqibang_;
  }

  if (seat_ != record.seat()) {
    // 鳴いた席以外のプレイヤ状態の変更は以上．
    return;
  }

  if (zimo_pai_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }

  // 副露牌を手牌から取り除く．
  for (std::uint_fast8_t i = 0u; i < record.tiles().size() - 1u; ++i) {
    std::uint_fast8_t const tile = Kanachan::pai2Num(record.tiles()[i]);
    bool found = false;
    for (std::uint_fast8_t j = hand_offset_[tile + 1]; j > hand_offset_[tile];) {
      --j;
      if (hand_[j] == 0u) {
        continue;
      }
      hand_[j] = 0u;
      found = true;
      break;
    }
    if (!found) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
  }
}

// 自摸牌を手牌に組み込む．
void PlayerState::handIn_()
{
  if (zimo_pai_ == std::numeric_limits<std::uint_fast8_t>::max()) {
    KANACHAN_THROW<std::invalid_argument>("");
  }

  std::uint_fast8_t i = hand_offsets_[zimo_pai_];
  for (; i < hand_offsets_[zimo_pai_ + 1u]; ++i) {
    if (hand_[i] == 1u) {
      continue;
    }
    if (hand_[i] != 0u) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    hand_[i] = 1u;
    break;
  }
  if (i == hand_offsets_[zimo_pai_ + 1u]) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max();
}

void PlayerState::onMyJiagang_(std::uint_fast8_t const tile)
{
  if (tile == zimo_pai_) {
    zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max();
  }
  else {
    {
      bool found = false;
      for (std::uint_fast8_t i = hand_offsets_[tile + 1u]; i > hand_offsets_[tile];) {
        --i;
        if (hand_[i] == 0u) {
          continue;
        }
        if (hand_[i] != 1u) {
          KANACHAN_THROW<std::runtime_error>("A broken data.");
        }
        hand_[i] = 0u;
        found = true;
        break;
      }
      if (!found) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
    }
    handIn_();
  }
}

void PlayerState::onMyAngang_(std::uint_fast8_t const tile)
{
  if (tile == 0u || tile == 10u || tile == 20u) {
    std::uint_fast8_t const other = tile + 5u;
    std::uint_fast8_t count = 0u;
    if (hand_[hand_offsets_[tile]] == 1u) {
      hand_[hand_offsets_[tile]] = 0u;
      ++count;
    }
    for (std::uint_fast8_t i = hand_offsets_[other + 1u]; i > hand_offsets_[other];) {
      --i;
      if (hand_[i] == 0u) {
        continue;
      }
      if (hand_[i] != 1u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      hand_[i] = 0u;
      ++count;
    }
    if (zimo_pai_ == tile || zimo_pai_ == other) {
      zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max();
      ++count;
    }
    else {
      handIn_();
    }
    if (count != 4u) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
  }
  else if (tile == 5u || tile == 15u || tile == 25u) {
    std::uint_fast8_t const other = tile - 5u;
    std::uint_fast8_t count = 0u;
    for (std::uint_fast8_t i = hand_offsets_[tile + 1u]; i > hand_offsets_[tile];) {
      --i;
      if (hand_[i] == 0u) {
        continue;
      }
      if (hand_[i] != 1u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      hand_[i] = 0u;
      ++count;
    }
    if (hand_[hand_offsets_[other]] == 1u) {
      hand_[hand_offsets_[other]] = 0u;
      ++count;
    }
    if (zimo_pai_ == tile || zimo_pai_ == other) {
      zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max();
      ++count;
    }
    else {
      handIn_();
    }
    if (count != 4u) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
  }
  else {
    std::uint_fast8_t count = 0u;
    for (std::uint_fast8_t i = hand_offsets_[tile + 1u]; i > hand_offsets_[tile];) {
      --i;
      if (hand_[i] == 0u) {
        continue;
      }
      if (hand_[i] != 1u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      hand_[i] = 0u;
      ++count;
    }
    if (zimo_pai_ == tile) {
      zimo_pai_ = std::numeric_limits<std::uint_fast8_t>::max();
      ++count;
    }
    else {
      handIn_();
    }
    if (count != 4u) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
  }
}

void PlayerState::onGang(lq::RecordAnGangAddGang const &record)
{
  if (record.doras().size() != 0u) {
    if (record.doras().size() >= 5u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << "A broken data: # of dora indicators = " << record.doras().size();
    }
    for (std::uint_fast8_t i = 0u; i < record.doras().size(); ++i) {
      dora_indicators_[i] = Kanachan::pai2Num(record.doras()[i]);
    }
  }

  if (record.seat() != seat_) {
    return;
  }

  if (record.tiles().size() != 2u) {
    KANACHAN_THROW<std::runtime_error>(_1) << "tiles = " << record.tiles();
  }
  std::uint_fast8_t const tile = Kanachan::pai2Num(record.tiles());

  if (record.type() == 2u) {
    // 加槓
    this->onMyJiagang_(tile);
  }
  else if (record.type() == 3u) {
    // 暗槓
    this->onMyAngang_(tile);
  }
  else {
    KANACHAN_THROW<std::runtime_error>(_1) << "A broken data: type = " << record.type();
  }
}

void PlayerState::onLiuju(
  lq::RecordLiuJu const &record, std::uint_fast8_t const seat,
  lq::RecordNewRound const &next_record)
{
  if (record.type() == 2u || record.type() == 3u) {
    // 四風連打または四槓散了
    if (seat >= 4u) {
      KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(seat);
    }
    if (record.has_liqi()) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    std::int_fast32_t const new_score = next_record.scores()[seat_];
    if (new_score != score_ && new_score != score_ - 1000) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << '(' << score_ << ", " << new_score << ')';
    }
    if (new_score == score_ - 1000) {
      // 次局開始時の点数が現在の点数よりも1000点減っているということは，
      // 四風連打もしくは四槓散了を確定させた打牌が立直宣言だったということ．
      // そしてこの立直宣言は成立する．
      if (seat == seat_) {
        score_ -= 1000;
      }
      ++liqibang_;
    }
  }
  else if (record.type() == 4u) {
    // 四家立直
    if (seat >= 4u) {
      KANACHAN_THROW<std::invalid_argument>(_1) << static_cast<unsigned>(seat);
    }
    if (record.has_liqi()) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    if (seat == seat_) {
      score_ -= 1000;
    }
    ++liqibang_;
  }
}

std::uint_fast8_t PlayerState::getSeat() const
{
  return seat_;
}

std::uint_fast8_t PlayerState::getLeftTileCount() const
{
  return count_;
}

std::uint_fast8_t PlayerState::getLevel() const
{
  return level_;
}

std::uint_fast8_t PlayerState::getRank(
  std::array<PlayerState, 4u> const &player_states) const
{
  std::uint_fast8_t rank = 0u;
  for (std::uint_fast8_t i = 0u; i < seat_; ++i) {
    if (player_states[i].getCurrentScore() >= score_) {
      ++rank;
    }
  }
  for (std::uint_fast8_t i = seat_ + 1u; i < 4u; ++i) {
    if (player_states[i].getCurrentScore() > score_) {
      ++rank;
    }
  }
  return rank;
}

std::int_fast32_t PlayerState::getInitialScore() const
{
  return initial_score_;
}

std::int_fast32_t PlayerState::getCurrentScore() const
{
  return score_;
}

std::vector<std::uint_fast8_t> PlayerState::getDiscardableTiles() const
{
  std::vector<std::uint_fast8_t> result;
  for (std::uint_fast8_t i = 0u; i < hand_offset_.size() - 1u; ++i) {
    std::uint_fast8_t const offset = hand_offset_[i];
    if (hand_[offset] == 1u) {
      result.push_back(i);
    }
  }
  return result;
}

std::uint_fast8_t PlayerState::getZimopai() const
{
  return zimo_pai_;
}

std::uint_fast8_t PlayerState::getGameRank() const
{
  return game_rank_;
}

std::int_fast32_t PlayerState::getGameScore() const
{
  return game_score_;
}

std::int_fast32_t PlayerState::getDeltaGradingPoint() const
{
  return delta_grading_point_;
}

void PlayerState::print(
  std::array<PlayerState, 4u> const &players, std::ostream &os) const
{
  std::uint_fast16_t offset = 0u;
  os << offset + room_;
  offset += 5u;
  os << ',' << offset + num_rounds_type_;
  offset += 2u;
  for (std::uint_fast8_t i = 0u; i < players.size(); ++i) {
    auto const &player = players[i];
    os << ',' << offset + player.getLevel();
    offset += 16u;
  }
  os << ',' << offset + seat_;
  offset += 4u;
  os << ',' << offset + chang_;
  offset += 3u;
  os << ',' << offset + ju_;
  offset += 4u;
  os << ',' << offset + count_;
  offset += 70u;
  os << ',' << offset + dora_indicators_[0u];
  offset += 37u;
  for (std::uint_fast8_t i = 1u; i < 5u; ++i) {
    if (dora_indicators_[i] != std::numeric_limits<std::uint_fast8_t>::max()) {
      os << ',' << offset + dora_indicators_[i];
    }
    offset += 37u;
  }
  for (auto const &h : hand_) {
    if (h == 1u) {
      os << ',' << offset;
    }
    ++offset;
  }
  if (zimo_pai_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    os << ',' << offset + zimo_pai_;
  }
  offset += 37u;
  if (offset != 510u) {
    KANACHAN_THROW<std::logic_error>(_1) << offset;
  }

  os << '\t';
  os << static_cast<unsigned>(ben_);
  os << ',' << static_cast<unsigned>(liqibang_);
  for (std::uint_fast8_t i = 0u; i < players.size(); ++i) {
    auto const &player = players[i];
    os << ',' << player.getCurrentScore();
  }
}

} // namespace Kanachan
