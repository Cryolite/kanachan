#include "annotation/annotation.hpp"
#include "annotation/round_progress.hpp"
#include "annotation/player_state.hpp"
#include "annotation/utility.hpp"
#include "common/throw.hpp"
#include "common/mahjongsoul.pb.h"
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

using RoundRankInput = std::array<std::pair<std::uint_fast8_t, std::int_fast32_t>, 4u>;

std::uint_fast8_t getRoundRank_(std::uint_fast8_t const seat, RoundRankInput data)
{
  std::sort(
    data.begin(), data.end(),
    [](auto const &lhs, auto const &rhs) -> bool{
      return lhs.second > rhs.second || lhs.second == rhs.second && lhs.first < rhs.first;
    });
  for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
    if (data[i].first == seat) {
      return i;
    }
  }
  KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
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
    // ????????????????????????????????????
    KANACHAN_THROW<std::runtime_error>(_1)
      << ph << ": category = " << config.category();
  }

  lq::GameMetaData const &meta_data = config.meta();
  std::uint_fast32_t const mode_id = meta_data.mode_id();

  std::array<std::uint_fast32_t, 4u> levels;
  {
    std::size_t i = 0u;
    for (auto const &account : header.accounts()) {
      levels[i] = account.level().id();
      ++i;
    }
  }

  std::array<std::uint_fast8_t, 4u> ranks = getRanks_(header.result());

  std::array<std::int_fast32_t, 4u> game_scores{
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max()
  };
  std::array<std::int_fast32_t, 4u> delta_grading_points{
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max(),
    std::numeric_limits<std::int_fast32_t>::max()
  };
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
      KANACHAN_THROW<std::runtime_error>(_1) << ph << ": a broken data.";
    }
  }
  for (auto delta_grading_point : delta_grading_points) {
    if (delta_grading_point == std::numeric_limits<std::int_fast32_t>::max()) {
      KANACHAN_THROW<std::runtime_error>(_1) << ph << ": a broken data.";
    }
  }

  wrapper.ParseFromString(msg0.data());
  if (wrapper.name() != ".lq.GameDetailRecords") {
    KANACHAN_THROW<std::runtime_error>(_1) << ph << ": a broken data.";
  }

  lq::GameDetailRecords msg1;
  msg1.ParseFromString(wrapper.data());

  std::array<Kanachan::PlayerState, 4u> player_states{
    Kanachan::PlayerState(0u, mode_id, levels[0u], ranks[0u], game_scores[0u], delta_grading_points[0u]),
    Kanachan::PlayerState(1u, mode_id, levels[1u], ranks[1u], game_scores[1u], delta_grading_points[1u]),
    Kanachan::PlayerState(2u, mode_id, levels[2u], ranks[2u], game_scores[2u], delta_grading_points[2u]),
    Kanachan::PlayerState(3u, mode_id, levels[3u], ranks[3u], game_scores[3u], delta_grading_points[3u])
  };

  std::string chang;
  std::size_t ju = std::numeric_limits<std::size_t>::max();
  std::size_t ben = std::numeric_limits<std::size_t>::max();

  std::uint_fast8_t prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
  std::uint_fast8_t prev_dapai = std::numeric_limits<std::uint_fast8_t>::max();

  std::array<std::vector<lq::OptionalOperation>, 4u> prev_action_candidates_list;

  Kanachan::RoundProgress round_progress;
  std::array<std::vector<Kanachan::Annotation>, 4u> player_annotations;

  std::uint_fast32_t const game_record_version = msg1.version();
  if (game_record_version == 0u) {
    // The version of game records before the maintenance on 2021/07/28 (JST).
  }
  else if (game_record_version == 210715u) {
    // The version of game records after the maintenance on 2021/07/28 (JST).
  }
  else {
    KANACHAN_THROW<std::runtime_error>(_1)
      << game_record_version << ": An unsupported game record version.";
  }

  auto const &records_0 = msg1.records();
  auto const &records_210715 = msg1.actions();
  std::size_t const record_size = [&]() -> std::size_t
  {
    if (game_record_version == 0u) {
      if (records_210715.size() != 0u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      return records_0.size();
    }
    if (game_record_version == 210715u) {
      if (records_0.size() != 0u) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      return records_210715.size();
    }
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }();

  for (std::size_t record_count = 0u; record_count < record_size; ++record_count) {
    std::string const &r = [&]() -> std::string const &
    {
      if (game_record_version == 0u) {
        return records_0[record_count];
      }
      if (game_record_version == 210715u) {
        return records_210715[record_count].result();
      }
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }();
    if (game_record_version == 210715u && r.empty()) {
      continue;
    }
    wrapper.ParseFromString(r);
    if (wrapper.name() == ".lq.RecordAnGangAddGang") {
      // ?????????????????????
      lq::RecordAnGangAddGang record;
      record.ParseFromString(wrapper.data());

      {
        std::uint_fast8_t const seat = record.seat();
        auto const &prev_action_candidates = prev_action_candidates_list[seat];
        if (prev_action_candidates.size() == 0u) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
        player_annotations[seat].emplace_back(
          seat, player_states, round_progress, prev_action_candidates, record);
      }

      for (auto &player_state : player_states) {
        player_state.onGang(record);
      }
      round_progress.onGang(record);

      // ????????????????????????????????????????????????????????????
      if (prev_dapai_seat != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      prev_dapai_seat = record.seat();
      prev_dapai = Kanachan::pai2Num(record.tiles());

      for (auto &prev_action_candidates : prev_action_candidates_list) {
        prev_action_candidates.clear();
      }
      for (auto const &operation : record.operations()) {
        std::uint_fast8_t const seat = operation.seat();
        if (seat == record.seat()) {
          KANACHAN_THROW<std::runtime_error>(_1) << uuid << ": a broken data.";
        }
        auto const &operation_list = operation.operation_list();
        prev_action_candidates_list[seat].assign(
          operation_list.cbegin(), operation_list.cend());
      }
    }
    else if (wrapper.name() == ".lq.RecordChiPengGang") {
      // ????????????????????????????????????
      lq::RecordChiPengGang record;
      record.ParseFromString(wrapper.data());

      {
        std::uint_fast8_t const seat = record.seat();
        for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
          auto const &prev_action_candidates = prev_action_candidates_list[i];
          if (i == seat && prev_action_candidates.size() == 0u) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
          if (prev_action_candidates.size() == 0u) {
            continue;
          }
          player_annotations[i].emplace_back(
            i, player_states, round_progress, prev_dapai_seat, prev_dapai,
            prev_action_candidates, record, i != seat);
        }
      }

      for (auto &player_state : player_states) {
        player_state.onChiPengGang(record);
      }
      round_progress.onChiPengGang(record);

      if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      if (prev_dapai == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
      prev_dapai = std::numeric_limits<std::uint_fast8_t>::max();

      for (auto &prev_action_candidates : prev_action_candidates_list) {
        prev_action_candidates.clear();
      }
      if (record.has_operation()) {
        auto const &operation = record.operation();
        std::uint_fast8_t const seat = operation.seat();
        if (record.type() == 0u || record.type() == 1u) {
          if (seat != record.seat()) {
            KANACHAN_THROW<std::runtime_error>(_1) << uuid << ": a broken data.";
          }
        }
        else {
          // ???????????????????????????????????????????????????
          KANACHAN_THROW<std::runtime_error>(_1) << uuid << ": a broken data.";
        }
        auto const &operation_list = operation.operation_list();
        prev_action_candidates_list[seat].assign(
          operation_list.cbegin(), operation_list.cend());
      }
    }
    else if (wrapper.name() == ".lq.RecordDealTile") {
      // ??????
      lq::RecordDealTile record;
      record.ParseFromString(wrapper.data());

      for (std::uint_fast8_t i = 0; i < 4u; ++i) {
        auto const &prev_action_candidates = prev_action_candidates_list[i];
        if (prev_action_candidates.size() == 0u) {
          continue;
        }
        player_annotations[i].emplace_back(
          i, player_states, round_progress, prev_dapai_seat, prev_dapai,
          prev_action_candidates, record);
      }

      try {
        for (auto &player_state : player_states) {
          player_state.onZimo(record);
        }
      } catch (...) {
        std::cerr << uuid << ": " << chang << ju << "???" << ben << "??????: "
                  << static_cast<unsigned>(player_states[0u].getLeftTileCount())
                  << std::endl;
        throw;
      }
      round_progress.onZimo(record);

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
      prev_dapai = std::numeric_limits<std::uint_fast8_t>::max();

      for (auto &prev_action_candidates : prev_action_candidates_list) {
        prev_action_candidates.clear();
      }
      if (record.has_operation()) {
        auto const &operation = record.operation();
        std::uint_fast8_t const seat = operation.seat();
        if (seat != record.seat()) {
          KANACHAN_THROW<std::runtime_error>(_1) << uuid << ": a broken data.";
        }
        auto const &operation_list = operation.operation_list();
        prev_action_candidates_list[seat].assign(
          operation_list.cbegin(), operation_list.cend());
      }
    }
    else if (wrapper.name() == ".lq.RecordDiscardTile") {
      // ??????
      lq::RecordDiscardTile record;
      record.ParseFromString(wrapper.data());

      {
        std::uint_fast8_t const seat = record.seat();
        auto const &prev_action_candidates = prev_action_candidates_list[seat];
        if (prev_action_candidates.size() > 0u) {
          try {
            player_annotations[seat].emplace_back(
              seat, player_states, round_progress, prev_action_candidates,
              record);
          } catch (...) {
            std::cerr << uuid << ": " << chang << ju << "???" << ben << "??????: "
                      << static_cast<unsigned>(player_states[seat].getLeftTileCount())
                      << std::endl;
            throw;
          }
        }
      }

      try {
        for (auto &player_state : player_states) {
          player_state.onDapai(record);
        }
      } catch (...) {
        std::cerr << uuid << ": " << chang << ju << "???" << ben << "??????: "
                  << static_cast<unsigned>(player_states[0u].getLeftTileCount())
                  << std::endl;
        throw;
      }
      round_progress.onDapai(record);

      if (prev_dapai_seat != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      if (prev_dapai != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      prev_dapai_seat = record.seat();
      prev_dapai = Kanachan::pai2Num(record.tile());

      for (auto &prev_action_candidates : prev_action_candidates_list) {
        prev_action_candidates.clear();
      }
      for (auto const &operation : record.operations()) {
        std::uint_fast8_t const seat = operation.seat();
        if (seat == record.seat()) {
          KANACHAN_THROW<std::runtime_error>(_1) << uuid << ": a broken data.";
        }
        auto const &operation_list = operation.operation_list();
        prev_action_candidates_list[seat].assign(
          operation_list.cbegin(), operation_list.cend());
      }
    }
    else if (wrapper.name() == ".lq.RecordHule") {
      // ??????
      lq::RecordHule record;
      record.ParseFromString(wrapper.data());

      for (std::uint_fast8_t i = 0; i < 4u; ++i) {
        lq::HuleInfo const *p_hule = nullptr;
        for (auto const &hule : record.hules()) {
          std::uint_fast8_t const seat = hule.seat();
          if (i == seat) {
            p_hule = &hule;
            break;
          }
        }
        auto const &prev_action_candidates = prev_action_candidates_list[i];
        if (p_hule != nullptr && prev_action_candidates.size() == 0u) {
          KANACHAN_THROW<std::logic_error>(_1) << "A logic error.";
        }
        if (prev_action_candidates.size() == 0u) {
          continue;
        }
        player_annotations[i].emplace_back(
          i, player_states, round_progress, prev_dapai_seat, prev_dapai,
          prev_action_candidates, p_hule);
      }

      std::array<std::int_fast32_t, 4u> const round_delta_scores{
        record.scores()[0u] - player_states[0u].getInitialScore(),
        record.scores()[1u] - player_states[1u].getInitialScore(),
        record.scores()[2u] - player_states[2u].getInitialScore(),
        record.scores()[3u] - player_states[3u].getInitialScore()
      };

      RoundRankInput const round_rank_input{
        std::make_pair(0u, record.scores()[0u]),
        std::make_pair(1u, record.scores()[1u]),
        std::make_pair(2u, record.scores()[2u]),
        std::make_pair(3u, record.scores()[3u])
      };
      std::array<std::uint_fast8_t, 4u> const round_ranks{
        getRoundRank_(0u, round_rank_input),
        getRoundRank_(1u, round_rank_input),
        getRoundRank_(2u, round_rank_input),
        getRoundRank_(3u, round_rank_input)
      };

      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        std::array<std::uint_fast8_t, 3u> results{
          std::numeric_limits<std::uint_fast8_t>::max(),
          std::numeric_limits<std::uint_fast8_t>::max(),
          std::numeric_limits<std::uint_fast8_t>::max()
        };
        for (std::uint_fast8_t j = 0u; auto const &hule : record.hules()) {
          if (j >= 3u) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
          if (results[j] != std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }

          if (hule.zimo()) {
            if (record.hules().size() != 1u) {
              KANACHAN_THROW<std::runtime_error>("A broken data.");
            }
            if (j != 0u) {
              KANACHAN_THROW<std::runtime_error>("A broken data.");
            }
            if (hule.seat() == i) {
              // ???????????????
              results[j++] = 0u;
            }
            else {
              // ???????????????:
              //   - 1u: ???????????????
              //   - 2u: ???????????????
              //   - 3u: ???????????????
              results[j++] = (4u + hule.seat() - i) % 4u;
            }
          }
          else {
            if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
              KANACHAN_THROW<std::logic_error>("A logic error.");
            }
            if (hule.seat() == i) {
              // ????????????:
              //   - 4u: ?????????????????????
              //   - 5u: ?????????????????????
              //   - 6u: ?????????????????????
              results[j++] = (4u + prev_dapai_seat - i) % 4u + 3u;
            }
            else if (prev_dapai_seat == i) {
              // ??????:
              //   - 7u: ??????????????????
              //   - 8u: ??????????????????
              //   - 9u: ??????????????????
              results[j++] = (4u + hule.seat() - i) % 4u + 6u;
            }
            else {
              // ?????????:
              //   - 10u: ??????????????????????????????
              //   - 11u: ??????????????????????????????
              //   - 12u: ??????????????????????????????
              //   - 13u: ??????????????????????????????
              //   - 14u: ??????????????????????????????
              //   - 15u: ??????????????????????????????
              if (hule.seat() == (i + 1u) % 4u) {
                // ??????????????????

                if (prev_dapai_seat == (i + 2u) % 4u) {
                  // ??????????????????????????????
                  results[j++] = 10u;
                }
                else if (prev_dapai_seat == (i + 3u) % 4u) {
                  // ??????????????????????????????
                  results[j++] = 11u;
                }
                else {
                  KANACHAN_THROW<std::logic_error>("A logic error.");
                }
              }
              else if (hule.seat() == (i + 2u) % 4u) {
                // ??????????????????

                if (prev_dapai_seat == (i + 1u) % 4u) {
                  // ??????????????????????????????
                  results[j++] = 12u;
                }
                else if (prev_dapai_seat == (i + 3u) % 4u) {
                  // ??????????????????????????????
                  results[j++] = 13u;
                }
                else {
                  KANACHAN_THROW<std::logic_error>(_1)
                    << "i = " << static_cast<unsigned>(i)
                    << ", hule.seat() = " << hule.seat()
                    << ", prev_dapai_seat = " << static_cast<unsigned>(prev_dapai_seat);
                }
              }
              else if (hule.seat() == (i + 3u) % 4u) {
                // ??????????????????

                if (prev_dapai_seat == (i + 1u) % 4u) {
                  // ??????????????????????????????
                  results[j++] = 14u;
                }
                else if (prev_dapai_seat == (i + 2u) % 4u) {
                  // ??????????????????????????????
                  results[j++] = 15u;
                }
                else {
                  KANACHAN_THROW<std::logic_error>("A logic error.");
                }
              }
              else {
                KANACHAN_THROW<std::runtime_error>("A broken data.");
              }
            }
          }
        }

        for (auto const &annotation : player_annotations[i]) {
          if (results[0u] == std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
          for (auto const &result : results) {
            if (result == std::numeric_limits<std::uint_fast8_t>::max()) {
              break;
            }
            annotation.printWithRoundResult(
              uuid, i, round_progress, result, round_delta_scores, round_ranks,
              std::cout);
          }
        }
        player_annotations[i].clear();
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
      prev_dapai = std::numeric_limits<std::uint_fast8_t>::max();

      for (auto &prev_action_candidates : prev_action_candidates_list) {
        prev_action_candidates.clear();
      }
    }
    else if (wrapper.name() == ".lq.RecordLiuJu") {
      // ????????????
      lq::RecordLiuJu record;
      record.ParseFromString(wrapper.data());

      if (record.type() == 1u) {
        // ????????????
        std::uint_fast8_t const seat = record.seat();
        auto const &prev_action_candidates = prev_action_candidates_list[seat];
        {
          bool found = false;
          for (auto const &prev_action_candidate : prev_action_candidates) {
            if (prev_action_candidate.type() == 10u) {
              found = true;
              break;
            }
          }
          if (!found) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
        }
        player_annotations[seat].emplace_back(
          seat, player_states, round_progress, prev_action_candidates, record);
      }

      if (record_count + 1u >= record_size) {
        KANACHAN_THROW<std::runtime_error>("A broken data.");
      }
      // ????????????????????????????????????????????????????????????????????????????????????????????????
      // ??????????????????????????????????????????????????????????????????????????? record
      // (????????? lq::RecordNewRound) ??????????????????????????????
      // (?????????????????????1000??????????????????????????????) ????????????????????????????????????
      lq::RecordNewRound next_record;
      {
        std::string const &rr = [&]() -> std::string const &
        {
          if (game_record_version == 0u) {
            return records_0[record_count + 1u];
          }
          if (game_record_version == 210715u) {
            for (std::size_t i = record_count + 1u; record_count < record_size; ++i) {
              std::string const &rr_ = records_210715[i].result();
              if (!rr_.empty()) {
                return rr_;
              }
            }
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }();
        lq::Wrapper next_wrapper;
        next_wrapper.ParseFromString(rr);
        if (next_wrapper.name() != ".lq.RecordNewRound") {
          KANACHAN_THROW<std::runtime_error>(_1) << next_wrapper.name();
        }
        next_record.ParseFromString(next_wrapper.data());
      }

      for (auto &player_state : player_states) {
        player_state.onLiuju(record, prev_dapai_seat, next_record);
      }

      std::array<std::int_fast32_t, 4u> const round_delta_scores{
        player_states[0u].getCurrentScore() - player_states[0u].getInitialScore(),
        player_states[1u].getCurrentScore() - player_states[1u].getInitialScore(),
        player_states[2u].getCurrentScore() - player_states[2u].getInitialScore(),
        player_states[3u].getCurrentScore() - player_states[3u].getInitialScore()
      };

      RoundRankInput const round_rank_input{
        std::make_pair(0, player_states[0].getCurrentScore()),
        std::make_pair(1, player_states[1].getCurrentScore()),
        std::make_pair(2, player_states[2].getCurrentScore()),
        std::make_pair(3, player_states[3].getCurrentScore()),
      };
      std::array<std::uint_fast8_t, 4u> const round_ranks{
        getRoundRank_(0u, round_rank_input),
        getRoundRank_(1u, round_rank_input),
        getRoundRank_(2u, round_rank_input),
        getRoundRank_(3u, round_rank_input),
      };

      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        for (auto const &annotation : player_annotations[i]) {
          annotation.printWithRoundResult(
            uuid, i, round_progress, 18u, round_delta_scores, round_ranks,
            std::cout);
        }
        player_annotations[i].clear();
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
      prev_dapai = std::numeric_limits<std::uint_fast8_t>::max();

      for (auto &prev_action_candidates : prev_action_candidates_list) {
        prev_action_candidates.clear();
      }
    }
    else if (wrapper.name() == ".lq.RecordNewRound") {
      // ??????
      lq::RecordNewRound record;
      record.ParseFromString(wrapper.data());

      switch (record.chang()) {
      case 0u:
        chang = "???";
        break;
      case 1u:
        chang = "???";
        break;
      case 2u:
        chang = "???";
        break;
      default:
        break;
      }
      ju = record.ju() + 1;
      ben = record.ben();

      for (auto const &annotations : player_annotations) {
        if (!annotations.empty()) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
      }

      try {
        for (auto &player_state : player_states) {
          player_state.onNewRound(record);
        }
      } catch (...) {
        std::cout << uuid << ": " << chang << ju << "???" << ben << "??????: "
                  << static_cast<unsigned>(player_states[0u].getLeftTileCount())
                  << std::endl;
        throw;
      }
      round_progress.onNewRound(record);

      if (prev_dapai_seat != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      if (prev_dapai != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }

      for (auto &prev_action_candidates : prev_action_candidates_list) {
        prev_action_candidates.clear();
      }
      if (record.has_operation()) {
        auto const &operation = record.operation();
        std::uint_fast8_t const seat = operation.seat();
        if (seat != record.ju()) {
          KANACHAN_THROW<std::runtime_error>(_1) << uuid << ": a broken data.";
        }
        auto const &operation_list = operation.operation_list();
        prev_action_candidates_list[seat].assign(
          operation_list.cbegin(), operation_list.cend());
      }
    }
    else if (wrapper.name() == ".lq.RecordNoTile") {
      // ????????????
      lq::RecordNoTile record;
      record.ParseFromString(wrapper.data());

      if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      if (prev_dapai == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }

      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        auto const &prev_action_candidates = prev_action_candidates_list[i];
        {
          bool found = false;
          for (auto const &prev_action_candidate : prev_action_candidates) {
            if (prev_action_candidate.type() == 9u) {
              found = true;
              break;
            }
          }
          if (!found) {
            continue;
          }
        }
        // ???????????????????????????????????????????????????????????????????????????
        player_annotations[i].emplace_back(
          i, player_states, round_progress, prev_dapai_seat, prev_dapai,
          prev_action_candidates, record);
      }

      std::array<std::int_fast32_t, 4u> scores{
        player_states[0u].getCurrentScore(),
        player_states[1u].getCurrentScore(),
        player_states[2u].getCurrentScore(),
        player_states[3u].getCurrentScore()
      };
      if (record.scores()[0u].delta_scores().size() != 0u) {
        for (auto const &score : record.scores()) {
          if (score.delta_scores().size() != 4u) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
            scores[i] += score.delta_scores()[i];
          }
        }
      }

      std::array<std::int_fast32_t, 4u> const round_delta_scores{
        scores[0u] - player_states[0u].getInitialScore(),
        scores[1u] - player_states[1u].getInitialScore(),
        scores[2u] - player_states[2u].getInitialScore(),
        scores[3u] - player_states[3u].getInitialScore()
      };

      RoundRankInput const round_rank_input{
        std::make_pair(0u, scores[0u]),
        std::make_pair(1u, scores[1u]),
        std::make_pair(2u, scores[2u]),
        std::make_pair(3u, scores[3u])
      };
      std::array<std::uint_fast8_t, 4u> const round_ranks{
        getRoundRank_(0u, round_rank_input),
        getRoundRank_(1u, round_rank_input),
        getRoundRank_(2u, round_rank_input),
        getRoundRank_(3u, round_rank_input)
      };

      for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
        std::array<std::uint_fast8_t, 3u> results{
          std::numeric_limits<std::uint_fast8_t>::max(),
          std::numeric_limits<std::uint_fast8_t>::max(),
          std::numeric_limits<std::uint_fast8_t>::max()
        };
        if (record.liujumanguan()) {
          for (std::uint_fast8_t j = 0u; auto const &score : record.scores()) {
            if (j > 3u) {
              KANACHAN_THROW<std::runtime_error>("A broken data.");
            }
            if (score.seat() == i) {
              // ????????????????????????????????????????????????
              results[j++] = 0u;
            }
            else {
              // ?????????????????????????????????????????????????????????
              results[j++] = (4u + score.seat() - i) % 4u;
            }
          }
        }
        else {
          if (record.players().size() != 4u) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          if (record.players()[i].tingpai()) {
            // ??????
            results[0u] = 17u;
          }
          else {
            // ??????
            results[0u] = 16u;
          }
        }

        for (auto const &annotation : player_annotations[i]) {
          if (results[0u] == std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
          for (auto const &result : results) {
            if (result == std::numeric_limits<std::uint_fast8_t>::max()) {
              break;
            }
            annotation.printWithRoundResult(
              uuid, i, round_progress, result, round_delta_scores, round_ranks,
              std::cout);
          }
        }
        player_annotations[i].clear();
      }

      prev_dapai_seat = std::numeric_limits<std::uint_fast8_t>::max();
      prev_dapai = std::numeric_limits<std::uint_fast8_t>::max();

      for (auto &prev_action_candidates : prev_action_candidates_list) {
        prev_action_candidates.clear();
      }
    }
    else {
      throw std::runtime_error("A broken data.");
    }
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
