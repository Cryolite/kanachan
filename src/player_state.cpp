#include "player_state.hpp"

#include "utility.hpp"
#include "throw.hpp"
#include "mahjongsoul.pb.h"
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
  for (auto &i : he_) {
    i = std::numeric_limits<std::uint_fast16_t>::max();
  }
  for (auto &i : fulus_) {
    i = std::numeric_limits<std::uint_fast8_t>::max();
  }

  if (4u <= seat) {
    KANACHAN_THROW<std::invalid_argument>(_1) << "seat = " << static_cast<unsigned>(seat);
  }

  switch (mode_id) {
  case 2u:
    // 段位戦・銅の間・四人東風戦
    room_ = 0u;
    dongfeng_ = 1u;
    banzuang_ = 0u;
    break;
  case 3u:
    // 段位戦・銅の間・四人半荘戦
    room_ = 0u;
    dongfeng_ = 0u;
    banzuang_ = 1u;
    break;
  case 5u:
    // 段位戦・銀の間・四人東風戦
    room_ = 1u;
    dongfeng_ = 1u;
    banzuang_ = 0u;
    break;
  case 6u:
    // 段位戦・銀の間・四人半荘戦
    room_ = 1u;
    dongfeng_ = 0u;
    banzuang_ = 1u;
    break;
  case 8u:
    // 段位戦・金の間・四人東風戦
    room_ = 2u;
    dongfeng_ = 1u;
    banzuang_ = 0u;
    break;
  case 9u:
    // 段位戦・金の間・四人半荘戦
    room_ = 2u;
    dongfeng_ = 0u;
    banzuang_ = 1u;
    break;
  case 11u:
    // 段位戦・玉の間・四人東風戦
    room_ = 3u;
    dongfeng_ = 1u;
    banzuang_ = 0u;
    break;
  case 12u:
    // 段位戦・玉の間・四人半荘戦
    room_ = 3u;
    dongfeng_ = 0u;
    banzuang_ = 1u;
    break;
  case 15u:
    // 段位戦・王座の間・四人東風戦
    room_ = 4u;
    dongfeng_ = 1u;
    banzuang_ = 0u;
    break;
  case 16u:
    // 段位戦・王座の間・四人半荘戦
    room_ = 4u;
    dongfeng_ = 0u;
    banzuang_ = 1u;
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
    // 四麻魂天
    level_ = 15u;
    break;
  default:
    KANACHAN_THROW<std::invalid_argument>(_1) << "level = " << level;
  }
}

void PlayerState::onNewRound(lq::RecordNewRound const &record)
{
  qijia_ = (4u - seat_) % 4u;

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
  for (std::uint_fast8_t i = 1u; i < 4u; ++i) {
    dora_indicators_[i] = std::numeric_limits<std::uint_fast8_t>::max();
  }

  if (record.left_tile_count() != 69u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << "A broken data: # of left tiles = " << record.left_tile_count();
  }
  count_ = record.left_tile_count();

  rank_ = getRank_(seat_, record.scores().cbegin(), record.scores().cend());

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

  first_zimo_ = 1u;
  lingshang_zimo_ = 0u;
  yifa_ = 0u;
  chipeng_ = std::numeric_limits<std::uint_fast8_t>::max();

  for (auto &i : he_) {
    i = std::numeric_limits<std::uint_fast16_t>::max();
  }
  for (auto &i : fulus_) {
    i = std::numeric_limits<std::uint_fast8_t>::max();
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

std::uint_fast8_t PlayerState::getFuluTail_() const
{
  std::uint_fast8_t result = 0u;
  for (; result < fulus_.size(); ++result) {
    auto const &fulu = fulus_[result];
    if (fulu == std::numeric_limits<std::uint_fast8_t>::max()) {
      break;
    }
  }
  return result;
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

  first_zimo_ = 0u;
  lingshang_zimo_ = 0u;

  if (record.is_liqi() || record.is_wliqi()) {
    yifa_ = 1u;
  }
  else {
    yifa_ = 0u;
  }

  {
    std::uint_fast8_t i = 0u;
    for (; i < 25u; ++i) {
      if (he_[i] == std::numeric_limits<std::uint_fast16_t>::max()) {
        break;
      }
    }
    if (i >= 24u) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    std::uint_fast16_t offset = 0;
    if (record.moqie()) {
      if (chipeng_ != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      offset += 37u;
    }
    if (record.is_liqi() || record.is_wliqi()) {
      if (chipeng_ != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      offset += 74u;
    }
    if (chipeng_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      if (offset != 0u) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      if (chipeng_ >= 4u) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      offset = 37u * (4u + chipeng_);
    }
    if (offset + dapai >= 37u * 8u) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    he_[i] = offset + dapai;
  }

  chipeng_ = std::numeric_limits<std::uint_fast8_t>::max();
}

namespace{

constexpr std::uint_fast64_t encodeChi_(
  std::uint_fast8_t const a, std::uint_fast8_t const b, std::uint_fast8_t const c)
{
  return (1ul << a) + (1ul << b) + (1ul << (c + 10u));
}

} // namespace *unnamed*

void PlayerState::onMyChi_(std::array<std::uint_fast8_t, 3u> const &tiles)
{
  std::uint_fast8_t const color = tiles[0u] / 10u;
  for (auto tile : tiles) {
    if (tile / 10u != color) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
  }

  std::array<std::uint_fast8_t, 3u> numbers{
    static_cast<std::uint_fast8_t>(tiles[0u] - color * 10u),
    static_cast<std::uint_fast8_t>(tiles[1u] - color * 10u),
    static_cast<std::uint_fast8_t>(tiles[2u] - color * 10u)
  };

  std::uint_fast8_t offset = std::numeric_limits<std::uint_fast8_t>::max();
  switch ((1ul << numbers[0u]) + (1ul << numbers[1u]) + (1ul << (numbers[2u] + 10u))) {
  case encodeChi_(2u, 3u, 1u):
    offset = 0u;
    break;
  case encodeChi_(1u, 3u, 2u):
    offset = 1u;
    break;
  case encodeChi_(3u, 4u, 2u):
    offset = 2u;
    break;
  case encodeChi_(1u, 2u, 3u):
    offset = 3u;
    break;
  case encodeChi_(2u, 4u, 3u):
    offset = 4u;
    break;
  case encodeChi_(4u, 5u, 3u):
    offset = 5u;
    break;
  case encodeChi_(4u, 0u, 3u):
    offset = 6u;
    break;
  case encodeChi_(2u, 3u, 4u):
    offset = 7u;
    break;
  case encodeChi_(3u, 5u, 4u):
    offset = 8u;
    break;
  case encodeChi_(3u, 0u, 4u):
    offset = 9u;
    break;
  case encodeChi_(5u, 6u, 4u):
    offset = 10u;
    break;
  case encodeChi_(0u, 6u, 4u):
    offset = 11u;
    break;
  case encodeChi_(3u, 4u, 5u):
    offset = 12u;
    break;
  case encodeChi_(3u, 4u, 0u):
    offset = 13u;
    break;
  case encodeChi_(4u, 6u, 5u):
    offset = 14u;
    break;
  case encodeChi_(4u, 6u, 0u):
    offset = 15u;
    break;
  case encodeChi_(6u, 7u, 5u):
    offset = 16u;
    break;
  case encodeChi_(6u, 7u, 0u):
    offset = 17u;
    break;
  case encodeChi_(4u, 5u, 6u):
    offset = 18u;
    break;
  case encodeChi_(4u, 0u, 6u):
    offset = 19u;
    break;
  case encodeChi_(5u, 7u, 6u):
    offset = 20u;
    break;
  case encodeChi_(0u, 7u, 6u):
    offset = 21u;
    break;
  case encodeChi_(7u, 8u, 6u):
    offset = 22u;
    break;
  case encodeChi_(5u, 6u, 7u):
    offset = 23u;
    break;
  case encodeChi_(0u, 6u, 7u):
    offset = 24u;
    break;
  case encodeChi_(6u, 8u, 7u):
    offset = 25u;
    break;
  case encodeChi_(8u, 9u, 7u):
    offset = 26u;
    break;
  case encodeChi_(6u, 7u, 8u):
    offset = 27u;
    break;
  case encodeChi_(7u, 9u, 8u):
    offset = 28u;
    break;
  case encodeChi_(7u, 8u, 9u):
    offset = 29u;
    break;
  default:
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }

  std::uint_fast8_t const i = getFuluTail_();
  if (i == fulus_.size()) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  fulus_[i] = color * 30u + offset;

  if (chipeng_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  chipeng_ = i;
}

void PlayerState::onMyPeng_(std::array<std::uint_fast8_t, 3u> const &tiles)
{
  std::uint_fast8_t const i = getFuluTail_();
  if (i == fulus_.size()) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  fulus_[i] = fulu_offsets_[1u] + tiles[2u];

  if (chipeng_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  chipeng_ = i;
}

void PlayerState::onMyDaminggang_(std::array<std::uint_fast8_t, 4u> const &tiles)
{
  std::uint_fast8_t const i = getFuluTail_();
  if (i == fulus_.size()) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  fulus_[i] = fulu_offsets_[2u] + tiles[3u];

  lingshang_zimo_ = 1u;
}

void PlayerState::onChiPengGang(lq::RecordChiPengGang const &record)
{
  if (record.has_liqi()) {
    std::uint_fast8_t liqi_seat = std::numeric_limits<std::uint_fast8_t>::max();
    if (record.froms().size() == 3u) {
      if (record.type() != 0u && record.type() != 1u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      liqi_seat = record.froms()[2u];
    }
    else if (record.froms().size() == 4u) {
      if (record.type() != 2u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      liqi_seat = record.froms()[3u];
    }

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

  first_zimo_ = 0u;
  if (lingshang_zimo_ != 0u) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  yifa_ = 0u;

  if (seat_ != record.seat()) {
    return;
  }

  if (record.type() == 0u) {
    if (zimo_pai_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    if (record.tiles().size() != 3u) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    std::array<std::uint_fast8_t, 3u> tiles{
      Kanachan::pai2Num(record.tiles()[0u]),
      Kanachan::pai2Num(record.tiles()[1u]),
      Kanachan::pai2Num(record.tiles()[2u])
    };
    onMyChi_(tiles);
  }
  else if (record.type() == 1u) {
    if (zimo_pai_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    if (record.tiles().size() != 3u) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    std::array<std::uint_fast8_t, 3u> tiles{
      Kanachan::pai2Num(record.tiles()[0u]),
      Kanachan::pai2Num(record.tiles()[1u]),
      Kanachan::pai2Num(record.tiles()[2u])
    };
    onMyPeng_(tiles);
  }
  else if (record.type() == 2u) {
    if (zimo_pai_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    if (record.tiles().size() != 4u) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    std::array<std::uint_fast8_t, 4u> tiles{
      Kanachan::pai2Num(record.tiles()[0u]),
      Kanachan::pai2Num(record.tiles()[1u]),
      Kanachan::pai2Num(record.tiles()[2u]),
      Kanachan::pai2Num(record.tiles()[3u])
    };
    onMyDaminggang_(tiles);
  }
  else {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
}

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

  std::uint_fast8_t i = 0u;
  if (tile == 0u || tile == 10u || tile == 20u) {
    for (; i < fulus_.size(); ++i) {
      auto const &fulu = fulus_[i];
      if (fulu == fulu_offsets_[1u] + tile) {
        break;
      }
      if (fulu == fulu_offsets_[1u] + tile + 5u) {
        break;
      }
    }
  }
  else if (tile == 5u || tile == 15u || tile == 25u) {
    for (; i < fulus_.size(); ++i) {
      auto const &fulu = fulus_[i];
      if (fulu == fulu_offsets_[1u] + tile) {
        break;
      }
      if (fulu == fulu_offsets_[1u] + tile - 5u) {
        break;
      }
    }
  }
  else {
    for (; i < fulus_.size(); ++i) {
      auto const &fulu = fulus_[i];
      if (fulu == fulu_offsets_[1u] + tile) {
        break;
      }
    }
  }
  if (i == fulus_.size()) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  fulus_[i] = fulu_offsets_[3u] + tile;
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

  std::uint_fast8_t const i = getFuluTail_();
  if (i == fulus_.size()) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  fulus_[i] = fulu_offsets_[4u] + tile;
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
  first_zimo_ = 0u;
  yifa_ = 0u;

  if (record.seat() != seat_) {
    return;
  }

  if (record.tiles().size() != 2u) {
    KANACHAN_THROW<std::runtime_error>(_1) << "tiles = " << record.tiles();
  }
  std::uint_fast8_t tile = Kanachan::pai2Num(record.tiles());

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

  lingshang_zimo_ = 1u;
  chipeng_ = std::numeric_limits<std::uint_fast8_t>::max();
}

std::int_fast32_t PlayerState::getInitialScore() const
{
  return initial_score_;
}

std::int_fast32_t PlayerState::getCurrentScore() const
{
  return score_;
}

} // namespace Kanachan
