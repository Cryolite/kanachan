#include "annotation/round_progress.hpp"

#include "annotation/utility.hpp"
#include "common/throw.hpp"
#include "common/mahjongsoul.pb.h"
#include <iostream>
#include <functional>
#include <stdexcept>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

namespace{

using std::placeholders::_1;

} // namespace *unnamed*

void RoundProgress::onNewRound(lq::RecordNewRound const &)
{
  std::uint_fast16_t const event = beginning_of_round_offset_;
  if (event >= zimo_offset_) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
  events_.assign(1u, event);
}

void RoundProgress::onZimo(lq::RecordDealTile const &record)
{
  std::uint_fast16_t const event = zimo_offset_ + record.seat();
  if (event >= dapai_offset_) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
  //events_.push_back(event);
}

void RoundProgress::onDapai(lq::RecordDiscardTile const &record)
{
  std::uint_fast16_t const seat = record.seat();
  std::uint_fast16_t const tile = Kanachan::pai2Num(record.tile());
  std::uint_fast16_t const moqie = record.moqie() ? 1 : 0;
  std::uint_fast16_t const liqi = record.is_liqi() || record.is_wliqi() ? 1 : 0;
  std::uint_fast16_t const event = dapai_offset_ + 37u * 2u * 2u * seat + 2u * 2u * tile + 2u * moqie + liqi;
  if (event >= chi_offset_) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
  events_.push_back(event);
}

void RoundProgress::onChiPengGang(lq::RecordChiPengGang const &record)
{
  if (record.type() == 0u) {
    // チー
    if (record.tiles().size() != 3u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << record.tiles().size() << ": A broken data.";
    }
    std::array<std::uint_fast8_t, 3u> tiles{
      Kanachan::pai2Num(record.tiles()[0u]),
      Kanachan::pai2Num(record.tiles()[1u]),
      Kanachan::pai2Num(record.tiles()[2u])
    };
    std::uint_fast16_t const encode
      = Kanachan::encodeChi(tiles.cbegin(), tiles.cend());
    std::uint_fast16_t const event = chi_offset_ + 90u * record.seat() + encode;
    if (event >= peng_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    events_.push_back(event);
  }
  else if (record.type() == 1u) {
    // ポン
    std::uint_fast16_t const seat = record.seat();
    std::uint_fast16_t const from = record.froms()[2u];
    std::uint_fast16_t const relative_from = (4u + from - seat) % 4u - 1u;
    if (record.tiles().size() != 3u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << record.tiles().size() << ": A broken data.";
    }
    std::array<std::uint_fast8_t, 3> tiles{
      Kanachan::pai2Num(record.tiles()[0u]),
      Kanachan::pai2Num(record.tiles()[1u]),
      Kanachan::pai2Num(record.tiles()[2u])
    };
    std::uint_fast16_t const encode
      = Kanachan::encodePeng(tiles.cbegin(), tiles.cend());
    std::uint_fast16_t const event
      = peng_offset_ + 3u * 40u * seat + 40u * relative_from + encode;
    if (event >= daminggang_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    events_.push_back(event);
  }
  else if (record.type() == 2u) {
    // 大明槓
    std::uint_fast16_t const seat = record.seat();
    std::uint_fast16_t const from = record.froms()[3u];
    std::uint_fast16_t const relative_from = (4u + from - seat) % 4u - 1u;
    std::uint_fast16_t const tile = Kanachan::pai2Num(record.tiles()[3u]);
    std::uint_fast16_t event = daminggang_offset_ + 3 * 37 * seat + 37 * relative_from + tile;
    if (event >= angang_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    events_.push_back(event);
  }
  else {
    KANACHAN_THROW<std::runtime_error>("A broken data.");
  }
}

void RoundProgress::onGang(lq::RecordAnGangAddGang const &record)
{
  if (record.type() == 2u) {
    // 加槓
    std::uint_fast16_t const tile = Kanachan::pai2Num(record.tiles());
    std::uint_fast16_t const event = jiagang_offset_ + 37 * record.seat() + tile;
    events_.push_back(event);
  }
  else if (record.type() == 3u) {
    // 暗槓
    std::uint_fast16_t tile = Kanachan::pai2Num(record.tiles());
    if (tile == 0u || tile == 10u || tile == 20u) {
      tile += 5u;
    }
    if (tile < 10u) {
      tile -= 1u;
    }
    else if (tile < 20u) {
      tile -= 2u;
    }
    else {
      tile -= 3u;
    }
    std::uint_fast16_t const event = angang_offset_ + 34 * record.seat() + tile;
    if (event >= jiagang_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    events_.push_back(event);
  }
  else {
    KANACHAN_THROW<std::runtime_error>(_1) << "A broken data: type = " << record.type();
  }
}

std::size_t RoundProgress::getSize() const
{
  return events_.size();
}

void RoundProgress::print(std::size_t size, std::ostream &os) const
{
  if (events_.empty()) {
    KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
  }
  if (size == 0u) {
    KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
  }

  os << events_[0u];
  for (std::size_t i = 1u; i < size; ++i) {
    os << ',' << events_[i];
  }
}

} // namespace Kanachan
