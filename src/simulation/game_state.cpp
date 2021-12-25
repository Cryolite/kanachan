#include "simulation/game_state.hpp"

#include "simulation/utility.hpp"
#include "common/assert.hpp"
#include "common/throw.hpp"
#include <boost/python/import.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/str.hpp>
#include <boost/python/long.hpp>
#include <boost/python/object.hpp>
#include <boost/python/handle.hpp>
#include <Python.h>
#include <functional>
#include <array>
#include <stdexcept>
#include <cstdint>


namespace Kanachan{

using std::placeholders::_1;
namespace python = boost::python;

GameState::GameState(
  std::uint_fast8_t const room, bool const dong_feng_zhan,
  std::array<Seat, 4u> const &seats)
  : room_(room),
    dong_feng_zhan_(dong_feng_zhan),
    seats_(seats)
{
  KANACHAN_ASSERT((room_ < 5u));
  for (auto [grade, model] : seats_) {
    KANACHAN_ASSERT((grade < 16u));
    KANACHAN_ASSERT((!model.is_none()));
  }
}

std::uint_fast8_t GameState::getRoom() const
{
  return room_;
}

bool GameState::isDongfengZhan() const
{
  return dong_feng_zhan_;
}

std::uint_fast8_t GameState::getPlayerGrade(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  return seats_[seat].first;
}

std::uint_fast8_t GameState::getChang() const
{
  KANACHAN_ASSERT(
    (dong_feng_zhan_ && chang_ < 2u || !dong_feng_zhan_ && chang_ < 3u));
  return chang_;
}

std::uint_fast8_t GameState::getJu() const
{
  KANACHAN_ASSERT((ju_ < 4u));
  return ju_;
}

std::uint_fast8_t GameState::getBenChang() const
{
  return ben_chang_;
}

std::uint_fast8_t GameState::getNumLizhiDeposits() const
{
  return lizhi_deposits_;
}

std::int_fast32_t GameState::getPlayerScore(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  return scores_[seat];
}

std::uint_fast8_t
GameState::getPlayerRanking(std::uint_fast8_t const seat) const
{
  KANACHAN_ASSERT((seat < 4u));
  std::int_fast32_t const score = scores_[seat];
  std::uint_fast8_t ranking = 0u;
  for (std::uint_fast8_t i = 0u; i < seat; ++i) {
    if (scores_[i] >= score) {
      ++ranking;
    }
  }
  for (std::uint_fast8_t i = seat + 1u; i < 4u; ++i) {
    if (scores_[i] > score) {
      ++ranking;
    }
  }
  return ranking;
}

std::uint_fast16_t GameState::selectAction(
  std::uint_fast8_t const seat, python::list features) const
{
  KANACHAN_ASSERT((seat < 4u));
  KANACHAN_ASSERT((!features.is_none()));
  KANACHAN_ASSERT((python::len(features) == 4));

  python::object model = seats_[seat].second;
  python::object action;
  try {
    action = model(features);
  }
  catch (python::error_already_set const &) {
    auto const [type, value, traceback] = [](){
      PyObject *p_type = nullptr;
      PyObject *p_value = nullptr;
      PyObject *p_traceback = nullptr;
      PyErr_Fetch(&p_type, &p_value, &p_traceback);
      python::object type_{python::handle<>(p_type)};
      python::object value_;
      if (p_value != nullptr) {
        value_ = python::object{python::handle<>(p_value)};
      }
      else {
        value_ = python::object();
      }
      python::object traceback_;
      if (p_traceback != nullptr) {
        traceback_ = python::object{python::handle<>(p_traceback)};
      }
      else {
        traceback_ = python::object();
      }
      return std::tuple(type_, value_, traceback_);
    }();

    python::object m = python::import("traceback");
    python::object o = m.attr("format_exception")(type, value, traceback);
    o = python::str("").attr("join")(o);
    python::extract<std::string> str(o);
    KANACHAN_ASSERT((str.check()));
    KANACHAN_THROW<std::runtime_error>(str());
  }
  python::extract<python::list> candidates(features[3]);
  KANACHAN_ASSERT((candidates.check()));
  python::list candidates_ = candidates();
  python::extract<python::long_> action_(action);
  if (!action_.check()) {
    KANACHAN_THROW<std::runtime_error>("A type error.");
  }
  if (candidates_.count(action_()) != 1) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << python::extract<long>(action_())() << ": An invalid action.";
  }
  return python::extract<long>(action_());
}

void GameState::onSuccessfulLizhi(std::uint_fast8_t const seat)
{
  KANACHAN_ASSERT((seat < 4u));
  KANACHAN_ASSERT((scores_[seat] >= 1000));
  scores_[seat] -= 1000;
  ++lizhi_deposits_;
}

void GameState::addPlayerScore(
  std::uint_fast8_t const seat, std::int_fast32_t const score)
{
  scores_[seat] += score;
}

void GameState::onLianzhuang(RoundEndStatus const round_end_status)
{
  ++ben_chang_;
  if (round_end_status == RoundEndStatus::hule) {
    lizhi_deposits_ = 0u;
  }
}

void GameState::onLunzhuang(RoundEndStatus const round_end_status)
{
  if (round_end_status == RoundEndStatus::liuju) {
    KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
  }

  if (++ju_ == 4u) {
    ++chang_;
    ju_ = 0u;
  }

  switch (round_end_status) {
  case RoundEndStatus::hule:
    ben_chang_ = 0u;
    lizhi_deposits_ = 0u;
    break;
  case RoundEndStatus::huangpai_pingju:
    ++ben_chang_;
    break;
  default:
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
}

} // namespace Kanachan
