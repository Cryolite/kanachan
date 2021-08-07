#include "post_zimo_action.hpp"
#include "player_state.hpp"
#include "throw.hpp"
#include "mahjongsoul.pb.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <cstddef>


namespace {

using std::placeholders::_1;

std::array<std::uint_fast8_t, 4u> getRanks_(lq::GameEndResult const &record)
{
  std::array<std::pair<std::uint_fast8_t, std::int_fast32_t>, 4u> tmp{
    std::pair(std::numeric_limits<std::uint_fast8_t>::max(),
              std::numeric_limits<std::int_fast32_t>::max()),
    std::pair(std::numeric_limits<std::uint_fast8_t>::max(),
              std::numeric_limits<std::int_fast32_t>::max()),
    std::pair(std::numeric_limits<std::uint_fast8_t>::max(),
              std::numeric_limits<std::int_fast32_t>::max()),
    std::pair(std::numeric_limits<std::uint_fast8_t>::max(),
              std::numeric_limits<std::int_fast32_t>::max())
  };
  std::array<std::uint_fast8_t, 4u> result{
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max(),
    std::numeric_limits<std::uint_fast8_t>::max()
  };
  for (auto const &player : record.players()) {
    std::uint_fast8_t const seat = player.seat();
    if (seat >= 4u) {
      KANACHAN_THROW<std::runtime_error>("A broken data.");
    }
    tmp[seat] = std::pair(seat, player.total_point());
  }
  std::sort(
    tmp.begin(), tmp.end(),
    [](auto const &lhs, auto const &rhs) -> bool{
      return lhs.second > rhs.second;
    });
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    for (std::uint_fast8_t j = 0u; j < 4u; ++j) {
      if (tmp[j].first == i) {
        result[i] = j;
        break;
      }
    }
    if (result[i] >= 4u) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
  }
  return result;
}

void convert(std::filesystem::path const &ph) {
  std::string data = [&ph]() {
    std::ifstream ifs(ph, std::ios_base::in | std::ios_base::binary);
    for (std::size_t i = 0; i < 3u; ++i) {
      std::ifstream::int_type const c = ifs.get();
      if (c == std::ifstream::traits_type::eof()) {
        KANACHAN_THROW<std::runtime_error>(_1) << ph << ": a broken file.";
      }
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return std::move(oss).str();
  }();

  lq::Wrapper wrapper;
  wrapper.ParseFromString(data);
  if (wrapper.name() != "") {
    KANACHAN_THROW<std::runtime_error>(wrapper.name());
  }

  lq::ResGameRecord msg0;
  msg0.ParseFromString(wrapper.data());

  lq::RecordGame const &header = msg0.head();
  std::string const uuid = header.uuid();

  lq::GameConfig const &config = header.config();
  if (config.category() != 2u) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << ph << ": category = " << config.category();
  }

  lq::GameMetaData const &meta_data = config.meta();
  std::uint_fast32_t const mode_id = meta_data.mode_id();
  switch (mode_id) {
  case 2u:
    // 段位戦・銅の間・四人東風戦
  case 6u:
    // 段位戦・銀の間・四人半荘戦
  case 8u:
    // 段位戦・金の間・四人東風戦
  case 9u:
    // 段位戦・金の間・四人半荘戦
  case 11u:
    // 段位戦・玉の間・四人東風戦
  case 12u:
    // 段位戦・玉の間・四人半荘戦
  case 15u:
    // 段位戦・王座の間・四人東風戦
  case 16u:
    // 段位戦・王座の間・四人半荘戦
    break;
  default:
    KANACHAN_THROW<std::runtime_error>(_1) << ph << ": mode_id = " << mode_id;
  }

  std::array<std::uint_fast32_t, 4u> levels;
  {
    std::size_t i = 0u;
    for (auto const &account : header.accounts()) {
      levels[i] = account.level().id();
      ++i;
    }
  }

  std::array<std::uint_fast8_t, 4u> ranks = getRanks_(header.result());

  std::array<std::int_fast32_t, 4u> game_scores;
  for (auto &e : game_scores) e = std::numeric_limits<std::int_fast32_t>::max();
  std::array<std::int_fast32_t, 4u> delta_grading_points;
  for (auto &e : delta_grading_points) e = std::numeric_limits<std::int_fast32_t>::max();
  for (std::uint_fast8_t i = 0u; i < 4u; ++i){
    for (auto const &player_game_result : header.result().players()) {
      if (player_game_result.seat() != i) {
        continue;
      }
      game_scores[i] = player_game_result.part_point_1();
      delta_grading_points[i] = player_game_result.grading_score();
    }
  }
  for (auto game_score : game_scores) {
    if (game_score == std::numeric_limits<std::int_fast32_t>::max()) {
      KANACHAN_THROW<std::runtime_error>(_1) << ph << ": a broken file.";
    }
  }
  for (auto delta_grading_point : delta_grading_points) {
    if (delta_grading_point == std::numeric_limits<std::int_fast32_t>::max()) {
      KANACHAN_THROW<std::runtime_error>(_1) << ph << ": a broken file.";
    }
  }

  wrapper.ParseFromString(msg0.data());
  if (wrapper.name() != ".lq.GameDetailRecords") {
    KANACHAN_THROW<std::runtime_error>(_1) << ph << ": a broken file.";
  }

  lq::GameDetailRecords msg1;
  msg1.ParseFromString(wrapper.data());

  std::array<Kanachan::PlayerState, 4u> states{
    Kanachan::PlayerState(0u, mode_id, levels[0u], ranks[0u], game_scores[0u], delta_grading_points[0u]),
    Kanachan::PlayerState(1u, mode_id, levels[1u], ranks[1u], game_scores[1u], delta_grading_points[1u]),
    Kanachan::PlayerState(2u, mode_id, levels[2u], ranks[2u], game_scores[2u], delta_grading_points[2u]),
    Kanachan::PlayerState(3u, mode_id, levels[3u], ranks[3u], game_scores[3u], delta_grading_points[3u])
  };

  std::array<std::vector<Kanachan::PostZimoAction>, 4u> post_zimo_actions;
  std::vector<Kanachan::PostZimoAction> results;

  std::string chang;
  std::size_t ju;
  std::size_t ben;

  std::uint_fast8_t prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();

  for (auto const &r : msg1.records()) {
    wrapper.ParseFromString(r);
    if (wrapper.name() == ".lq.RecordAnGangAddGang") {
      lq::RecordAnGangAddGang record;
      record.ParseFromString(wrapper.data());

      {
        std::uint_fast8_t const seat = record.seat();
        post_zimo_actions[seat].emplace_back(states, record);
      }

      for(auto &state : states) {
        state.onGang(record);
      }

      prev_dapai_seat = record.seat();
    }
    else if (wrapper.name() == ".lq.RecordChiPengGang") {
      lq::RecordChiPengGang record;
      record.ParseFromString(wrapper.data());

      for (auto &state : states) {
        state.onChiPengGang(record);
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
    }
    else if (wrapper.name() == ".lq.RecordDealTile") {
      lq::RecordDealTile record;
      record.ParseFromString(wrapper.data());

      try {
        for (auto &state : states) {
          state.onZimo(record);
        }
      } catch (...) {
        std::cout << uuid << ": " << chang << ju << "局" << ben << "本場" << std::endl;
        throw;
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
    }
    else if (wrapper.name() == ".lq.RecordDiscardTile") {
      lq::RecordDiscardTile record;
      record.ParseFromString(wrapper.data());

      {
        std::uint_fast8_t const seat = record.seat();
        post_zimo_actions[seat].emplace_back(states, record);
      }

      try {
        for (auto &state : states) {
          state.onDapai(record);
        }
      } catch (...) {
        std::cout << uuid << ": " << chang << ju << "局" << ben << "本場" << std::endl;
        throw;
      }

      prev_dapai_seat = record.seat();
    }
    else if (wrapper.name() == ".lq.RecordHule") {
      lq::RecordHule record;
      record.ParseFromString(wrapper.data());

      if (record.hules().size() == 1u && record.hules()[0u].zimo()) {
        auto const &hule = record.hules()[0u];
        std::uint_fast8_t const seat = hule.seat();
        post_zimo_actions[seat].emplace_back(states, hule);
      }

      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        std::int_fast32_t const delta_round_score = record.scores()[i] - states[i].getInitialScore();
        std::uint_fast8_t type = std::numeric_limits<std::uint_fast8_t>::max();
        for (auto const &hule : record.hules()) {
          if (hule.zimo()) {
            if (record.hules().size() != 1u) {
              KANACHAN_THROW<std::runtime_error>("A broken data.");
            }
            if (type != std::numeric_limits<std::uint_fast8_t>::max()) {
              KANACHAN_THROW<std::runtime_error>("A broken data.");
            }
            if (hule.seat() == i) {
              // 自摸和
              type = 0u;
            }
            else {
              // 他家自摸和
              type = 1u;
            }
          }
          else {
            if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
              KANACHAN_THROW<std::logic_error>("A logic error.");
            }
            if (hule.seat() == i) {
              // 栄和
              // ダブロン・トリプルロンは自家の栄和とする．
              type = 2u;
            }
            else if (prev_dapai_seat == i) {
              if (type == std::numeric_limits<std::uint_fast8_t>::max() || type == 2u || type == 3u) {
                // 放銃
                type = 3u;
              }
              else {
                KANACHAN_THROW<std::runtime_error>("A broken data.");
              }
            }
            else {
              if (type == std::numeric_limits<std::uint_fast8_t>::max() || type == 2u || type == 4u) {
                // 横移動．ただし自家の包を含む．
                type = 4u;
              }
              else {
                KANACHAN_THROW<std::runtime_error>("A broken data.");
              }
            }
          }
        }
        for (auto &post_zimo_action : post_zimo_actions[i]) {
          switch (type) {
          case 0u:
            post_zimo_action.onZimohu(delta_round_score);
            break;
          case 1u:
            post_zimo_action.onTajiahu(delta_round_score);
            break;
          case 2u:
            post_zimo_action.onRong(delta_round_score);
            break;
          case 3u:
            post_zimo_action.onFangchong(delta_round_score);
            break;
          case 4u:
            post_zimo_action.onOther(delta_round_score);
            break;
          default:
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
          results.push_back(post_zimo_action);
        }
        post_zimo_actions[i].clear();
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
    }
    else if (wrapper.name() == ".lq.RecordLiuJu") {
      lq::RecordLiuJu record;
      record.ParseFromString(wrapper.data());

      if (record.type() == 1u) {
        std::uint_fast8_t const seat = record.seat();
        post_zimo_actions[seat].emplace_back(states, record);
      }

      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        std::int_fast32_t const delta_round_score = states[i].getCurrentScore() - states[i].getInitialScore();
        for (auto &post_zimo_action : post_zimo_actions[i]) {
          post_zimo_action.onLiuju(delta_round_score);
          results.push_back(post_zimo_action);
        }
        post_zimo_actions[i].clear();
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
    }
    else if (wrapper.name() == ".lq.RecordNewRound") {
      lq::RecordNewRound record;
      record.ParseFromString(wrapper.data());
      switch (record.chang()) {
      case 0u:
        chang = "東";
        break;
      case 1u:
        chang = "南";
        break;
      case 2u:
        chang = "西";
        break;
      default:
        break;
      }
      ju = record.ju() + 1;
      ben = record.ben();

      for (auto const &i : post_zimo_actions) {
        if (!i.empty()) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
      }

      try {
        for (auto &state : states) {
          state.onNewRound(record);
        }
      } catch (...) {
        std::cout << uuid << ": " << chang << ju << "局" << ben << "本場" << std::endl;
        throw;
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
    }
    else if (wrapper.name() == ".lq.RecordNoTile") {
      lq::RecordNoTile record;
      record.ParseFromString(wrapper.data());

      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        auto const &score = record.scores()[0u];
        std::int_fast32_t delta_round_score = score.old_scores()[i] - states[i].getInitialScore();
        if (score.delta_scores().size() != 0u) {
          if (score.delta_scores().size() != 4u) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          delta_round_score += score.delta_scores()[i];
        }

        std::uint_fast8_t type = std::numeric_limits<std::uint_fast8_t>::max();
        if (record.liujumanguan()) {
          bool found = false;
          for (auto const &score : record.scores()) {
            if (score.seat() == i) {
              type = 2u;
              found = true;
              break;
            }
          }
          if (!found) {
            type = 3u;
          }
        }
        else {
          if (record.players().size() != 4u) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          if (record.players()[i].tingpai()) {
            type = 1u;
          }
          else {
            type = 0u;
          }
        }

        for (auto &post_zimo_action : post_zimo_actions[i]) {
          switch (type) {
          case 0u:
            post_zimo_action.onButing(delta_round_score);
            break;
          case 1u:
            post_zimo_action.onTingpai(delta_round_score);
            break;
          case 2u:
            post_zimo_action.onLiujumanguan(delta_round_score);
            break;
          case 3u:
            post_zimo_action.onTajiahu(delta_round_score);
            break;
          default:
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
          results.push_back(post_zimo_action);
        }
        post_zimo_actions[i].clear();
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
    }
    else {
      throw std::runtime_error("A broken data.");
    }
  }

  for (auto const &i : results) {
    i.encode(uuid, std::cout);
    std::cout << '\n';
  }
}

} // namespace *unnamed*

int main(int const argc, char const *argv[]) {
  if (argc < 2) {
    KANACHAN_THROW<std::runtime_error>(_1) << "argc = " << argc;
  }

  std::filesystem::path ph(argv[1]);
  if (!std::filesystem::exists(ph)) {
    KANACHAN_THROW<std::invalid_argument>(std::placeholders::_1)
      << ph << ": does not exist.";
  }
  if (!std::filesystem::is_directory(ph)) {
    KANACHAN_THROW<std::invalid_argument>(std::placeholders::_1)
      << ph << ": not a directory.";
  }

  {
    auto const end = std::filesystem::directory_iterator();
    for (std::filesystem::directory_iterator iter(ph); iter != end; ++iter) {
      if (iter->is_regular_file()) {
        convert(iter->path());
      }
    }
  }

  return EXIT_SUCCESS;
}
