#include "post_zimo_action.hpp"

#include "player_state.hpp"
#include "utility.hpp"
#include "throw.hpp"
#include <iostream>
#include <array>
#include <functional>
#include <cstdint>


namespace Kanachan{

namespace{

using std::placeholders::_1;

} // *namespace*

namespace Detail_{

class Serializer
{
public:
  explicit Serializer(std::string const &uuid)
    : uuid_(uuid),
      sparse_binaries_(),
      sparse_binary_index_(0u),
      floats_(),
      action_(std::numeric_limits<std::uint_fast8_t>::max()),
      action_index_(0u),
      round_result_(std::numeric_limits<std::uint_fast8_t>::max()),
      delta_round_score_(),
      game_rank_(),
      game_score_(),
      delta_grading_point_()
  {}

  Serializer(Serializer const &) = delete;

  Serializer &operator=(Serializer const &) = delete;

  void pushSparseBinary(bool b)
  {
    if (b) {
      sparse_binaries_.push_back(sparse_binary_index_);
    }
    ++sparse_binary_index_;
  }

  template<typename T, typename U>
  void pushCategory(T const &num_categories, U const &category)
  {
    static_assert(std::numeric_limits<T>::max() >= std::numeric_limits<U>::max());
    if (num_categories < 0) {
      KANACHAN_THROW<std::invalid_argument>(_1) << "# of categories = " << num_categories;
    }
    if (category == std::numeric_limits<U>::max()) {
      sparse_binary_index_ += num_categories;
      return;
    }
    if (category >= num_categories) {
      KANACHAN_THROW<std::invalid_argument>(_1)
        << "category = " << category << ", # of categories = " << num_categories;
    }

    sparse_binary_index_ += category;
    sparse_binaries_.push_back(sparse_binary_index_);
    sparse_binary_index_ += num_categories - category;
  }

  void pushFloat(double const &val)
  {
    floats_.push_back(val);
  }

  void setActionCategory(std::size_t num_categories, std::uint_fast8_t category)
  {
    if (category == std::numeric_limits<std::uint_fast8_t>::max()) {
      action_index_ += num_categories;
      return;
    }
    if (action_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    action_index_ += category;
    action_ = action_index_;
    action_index_ += num_categories - category;
  }

  void setActionBinary(bool b)
  {
    if (b) {
      if (action_ != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      action_ = action_index_;
    }
    ++action_index_;
  }

  void setResult(
    std::uint_fast8_t round_result,
    std::int_fast32_t delta_round_score,
    std::uint_fast8_t game_rank,
    std::int_fast32_t game_score,
    std::int_fast32_t delta_grading_point)
  {
    round_result_ = round_result;
    delta_round_score_ = delta_round_score;
    game_rank_ = game_rank;
    game_score_ = game_score;
    delta_grading_point_ = delta_grading_point;
  }

  void encode(std::ostream &os)
  {
    if (action_ == std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }

    os << uuid_ << '\t';
    if (!sparse_binaries_.empty()) {
      os << sparse_binaries_.front();
      for (std::size_t i = 1u; i < sparse_binaries_.size(); ++i) {
        os << ',' << sparse_binaries_[i];
      }
    }
    os << '\t';
    if (!floats_.empty()) {
      os << floats_.front();
      for (std::size_t i = 1u; i < floats_.size(); ++i) {
        os << ',' << floats_[i];
      }
    }
    os << '\t' << static_cast<unsigned>(action_);
    os << '\t' << static_cast<unsigned>(round_result_)
       << ',' << delta_round_score_
       << ',' << static_cast<unsigned>(game_rank_)
       << ',' << game_score_
       << ',' << delta_grading_point_;
  }

private:
  std::string uuid_;
  std::vector<std::uint_fast16_t> sparse_binaries_;
  std::uint_fast16_t sparse_binary_index_;
  std::vector<double> floats_;
  std::uint_fast8_t action_;
  std::uint_fast8_t action_index_;
  std::uint_fast8_t round_result_;
  std::int_fast32_t delta_round_score_;
  std::uint_fast8_t game_rank_;
  std::int_fast32_t game_score_;
  std::int_fast32_t delta_grading_point_;
}; // class Serializer

} // namespace Detail_

PostZimoAction::Tajia_::Tajia_()
{}

void PostZimoAction::Tajia_::initialize(
  std::uint_fast8_t level,
  std::uint_fast8_t rank,
  std::int_fast32_t score,
  std::uint_fast8_t yifa,
  std::array<std::uint_fast16_t, 24u> const &he,
  std::array<std::uint_fast8_t, 4u> const &fulus)
{
  level_ = level;
  rank_ = rank;
  score_ = score;
  yifa_ = yifa;
  he_ = he;
  fulus_ = fulus;
}

void PostZimoAction::Tajia_::encode(Detail_::Serializer &s) const
{
  s.pushCategory(16u, level_);
  s.pushCategory(4u, rank_);
  s.pushFloat(score_);
  s.pushSparseBinary(yifa_ == 1u);
  for (auto const &i : he_) {
    s.pushCategory(37ul * 8ul, i);
  }
  for (auto const &i : fulus_) {
    s.pushCategory(238u, i);
  }
}

PostZimoAction::PostZimoAction(
  std::array<Kanachan::PlayerState, 4u> const &states,
  std::uint_fast8_t const seat)
{
  Kanachan::PlayerState const &state = states[seat];
  room_ = state.room_;
  dongfeng_ = state.dongfeng_;
  banzuang_ = state.banzuang_;
  qijia_ = state.qijia_;
  chang_ = state.chang_;
  ju_ = state.ju_;
  ben_ = state.ben_;
  liqibang_ = state.liqibang_;
  dora_indicators_ = state.dora_indicators_;
  count_ = state.count_;
  level_ = state.level_;
  rank_ = state.rank_;
  score_ = state.score_;
  hand_ = state.hand_;
  zimo_pai_ = state.zimo_pai_;
  first_zimo_ = state.first_zimo_;
  lingshang_zimo_ = state.lingshang_zimo_;
  yifa_ = state.yifa_;
  he_ = state.he_;
  fulus_ = state.fulus_;

  for (std::uint_fast8_t i = 0u; i < 3u; ++i) {
    auto const &state = states[(seat + i + 1u) % 4u];
    auto &tajia = tajias_[i];
    tajia.initialize(
      state.level_, state.rank_, state.score_, state.yifa_, state.he_, state.fulus_);
  }

  game_rank_ = state.game_rank_;
  game_score_ = state.game_score_;
  delta_grading_point_ = state.delta_grading_point_;
}

PostZimoAction::PostZimoAction(
  std::array<Kanachan::PlayerState, 4u> const &states,
  lq::RecordDiscardTile const &record)
  : PostZimoAction(states, record.seat())
{
  auto const &state = states[record.seat()];

  if (record.doras().size() != 0u) {
    for (std::uint_fast8_t i = 0u; i < record.doras().size(); ++i) {
      dora_indicators_[i] = Kanachan::pai2Num(record.doras()[i]);
    }
  }

  std::uint_fast8_t const dapai = Kanachan::pai2Num(record.tile());
  std::uint_fast8_t offset = 0u;
  if (record.moqie()) {
    if (state.chipeng_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    offset += 37u;
  }
  if (record.is_liqi() || record.is_wliqi()) {
    if (state.chipeng_ != std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    offset += 74u;
  }
  if (offset + dapai >= 37u * 4u) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
  dapai_ = offset + dapai;

  gang_ = std::numeric_limits<std::uint_fast8_t>::max();
  hule_ = 0u;
  liuju_ = 0u;
}

PostZimoAction::PostZimoAction(
  std::array<Kanachan::PlayerState, 4u> const &states,
  lq::RecordAnGangAddGang const &record)
  : PostZimoAction(states, record.seat())
{
  auto const &state = states[record.seat()];

  if (record.doras().size() != 0u) {
    for (std::uint_fast8_t i = 0u; i < record.doras().size(); ++i) {
      dora_indicators_[i] = Kanachan::pai2Num(record.doras()[i]);
    }
  }

  dapai_ = std::numeric_limits<std::uint_fast8_t>::max();

  if (record.tiles().size() != 2u) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  std::uint_fast8_t const tile = Kanachan::pai2Num(record.tiles());
  if (tile >= 37u) {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
  if (record.type() == 2u) {
    gang_ = tile;
  }
  else if (record.type() == 3u) {
    gang_ = 37u + tile;
  }
  else {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }

  hule_ = 0u;
  liuju_ = 0u;
}

PostZimoAction::PostZimoAction(
  std::array<Kanachan::PlayerState, 4u> const &states,
  lq::HuleInfo const &record)
  : PostZimoAction(states, record.seat())
{
  dapai_ = std::numeric_limits<std::uint_fast8_t>::max();
  gang_ = std::numeric_limits<std::uint_fast8_t>::max();
  hule_ = 1u;
  liuju_ = 0u;
}

PostZimoAction::PostZimoAction(
  std::array<Kanachan::PlayerState, 4u> const &states,
  lq::RecordLiuJu const &record)
  : PostZimoAction(states, record.seat())
{
  dapai_ = std::numeric_limits<std::uint_fast8_t>::max();
  gang_ = std::numeric_limits<std::uint_fast8_t>::max();
  hule_ = 0u;
  liuju_ = 1u;
}

void PostZimoAction::setRoundResult_(
  std::uint_fast8_t const result, std::int_fast32_t const delta_round_score)
{
  if (round_result_ != std::numeric_limits<std::uint_fast8_t>::max()) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
  if (delta_round_score_ != std::numeric_limits<std::int_fast32_t>::max()) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
  round_result_ = result;
  delta_round_score_ = delta_round_score;
}

void PostZimoAction::onZimohu(std::int_fast32_t delta_round_score)
{
  setRoundResult_(0u, delta_round_score);
}

void PostZimoAction::onTajiahu(std::int_fast32_t delta_round_score)
{
  setRoundResult_(1u, delta_round_score);
}

void PostZimoAction::onRong(std::int_fast32_t delta_round_score)
{
  setRoundResult_(2u, delta_round_score);
}

void PostZimoAction::onFangchong(std::int_fast32_t delta_round_score)
{
  setRoundResult_(3u, delta_round_score);
}

void PostZimoAction::onOther(std::int_fast32_t delta_round_score)
{
  setRoundResult_(4u, delta_round_score);
}

void PostZimoAction::onButing(std::int_fast32_t delta_round_score)
{
  setRoundResult_(5u, delta_round_score);
}

void PostZimoAction::onTingpai(std::int_fast32_t delta_round_score)
{
  setRoundResult_(6u, delta_round_score);
}

void PostZimoAction::onLiujumanguan(std::int_fast32_t delta_round_score)
{
  setRoundResult_(7u, delta_round_score);
}

void PostZimoAction::onLiuju(std::int_fast32_t delta_round_score)
{
  setRoundResult_(8u, delta_round_score);
}

void PostZimoAction::encode(std::string const &uuid, std::ostream &os) const
{
  Detail_::Serializer s(uuid);
  s.pushCategory(5u, room_);
  s.pushSparseBinary(dongfeng_ == 1u);
  s.pushSparseBinary(banzuang_ == 1u);
  s.pushCategory(4u, qijia_);
  s.pushCategory(3u, chang_);
  s.pushCategory(4u, ju_);
  s.pushFloat(ben_);
  s.pushFloat(liqibang_);
  for (auto const &i : dora_indicators_) {
    s.pushCategory(37u, i);
  }
  s.pushCategory(70u, count_);
  s.pushCategory(16u, level_);
  s.pushCategory(4u, rank_);
  s.pushFloat(score_);
  for (auto const &i : hand_) {
    s.pushSparseBinary(i == 1u);
  }
  s.pushCategory(37u, zimo_pai_);
  s.pushSparseBinary(first_zimo_ == 1u);
  s.pushSparseBinary(lingshang_zimo_ == 1u);
  s.pushSparseBinary(yifa_ == 1u);
  for (auto const &i : he_) {
    s.pushCategory(37ul * 8ul, i);
  }
  for (auto const &i : fulus_) {
    s.pushCategory(238u, i);
  }
  for (auto const &tajia : tajias_) {
    tajia.encode(s);
  }
  s.setActionCategory(37u * 4u, dapai_);
  s.setActionCategory(74u, gang_);
  s.setActionBinary(hule_ == 1u);
  s.setActionBinary(liuju_ == 1u);
  s.setResult(round_result_, delta_round_score_, game_rank_, game_score_, delta_grading_point_);
  s.encode(os);
}

} // namespace Kanachan
