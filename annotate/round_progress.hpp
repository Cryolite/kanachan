#if !defined(KANACHAN_ROUND_PROGRESS_HPP_INCLUDE_GUARD)
#define KANACHAN_ROUND_PROGRESS_HPP_INCLUDE_GUARD

#include "mahjongsoul.pb.h"
#include <iosfwd>
#include <vector>
#include <cstdint>
#include <cstddef>


namespace Kanachan{

class RoundProgress
{
public:
  RoundProgress() = default;

  RoundProgress(RoundProgress const &) = delete;

  RoundProgress &operator=(RoundProgress const &) = delete;

  void onNewRound(lq::RecordNewRound const &);

  void onZimo(lq::RecordDealTile const &record);

  void onDapai(lq::RecordDiscardTile const &record);

  void onChiPengGang(lq::RecordChiPengGang const &record);

  void onGang(lq::RecordAnGangAddGang const &record);

  std::size_t getSize() const;

  void print(std::size_t size, std::ostream &os) const;

private:
  static constexpr std::uint_fast16_t beginning_of_round_offset_ =    0u;
  static constexpr std::uint_fast16_t zimo_offset_               =    1u;
  static constexpr std::uint_fast16_t dapai_offset_              =    5u;
  static constexpr std::uint_fast16_t chi_offset_                =  597u;
  static constexpr std::uint_fast16_t peng_offset_               =  957u;
  static constexpr std::uint_fast16_t daminggang_offset_         = 1401u;
  static constexpr std::uint_fast16_t angang_offset_             = 1845u;
  static constexpr std::uint_fast16_t jiagang_offset_            = 1981u;

  std::vector<std::uint_fast16_t> events_;
}; // class RoundProgress

} // namespace Kanachan

#endif // !defined(KANACHAN_ROUND_PROGRESS_HPP_INCLUDE_GUARD)
