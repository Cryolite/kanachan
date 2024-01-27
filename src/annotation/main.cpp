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
#include <algorithm>
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
    // 段位戦のデータではない．
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
      // 暗槓または加槓
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

      // 槍槓があるので加槓・暗槓は打牌とみなす．
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
      // チー，ポン，または大明槓
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
          // 大明槓．選択肢が出る可能性は無い．
          KANACHAN_THROW<std::runtime_error>(_1) << uuid << ": a broken data.";
        }
        auto const &operation_list = operation.operation_list();
        prev_action_candidates_list[seat].assign(
          operation_list.cbegin(), operation_list.cend());
      }
    }
    else if (wrapper.name() == ".lq.RecordDealTile") {
      // 自摸
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
        std::cerr << uuid << ": " << chang << ju << "局" << ben << "本場: "
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
      // 打牌
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
            std::cerr << uuid << ": " << chang << ju << "局" << ben << "本場: "
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
        std::cerr << uuid << ": " << chang << ju << "局" << ben << "本場: "
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
      // 和了
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

      std::array<std::uint_fast8_t, 7u> results{
        UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX,
        UINT_FAST8_MAX, UINT_FAST8_MAX
      };
      for (std::uint_fast8_t j = 0u; auto const &hule : record.hules()) {
        if (j >= 3u) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
        if (results[j] != UINT_FAST8_MAX) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }

        if (hule.zimo()) {
          if (record.hules().size() != 1u) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          if (j != 0u) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          results[j++] = hule.seat();
        }
        else {
          if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
          if (prev_dapai_seat == hule.seat()) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          constexpr std::array<std::array<std::uint_fast8_t, 4u>, 4u> codes{
            std::array<std::uint_fast8_t, 4u>{ UINT_FAST8_MAX, 4u, 5u, 6u },
            std::array<std::uint_fast8_t, 4u>{ 7u, UINT_FAST8_MAX, 8u, 9u },
            std::array<std::uint_fast8_t, 4u>{ 10u, 11u, UINT_FAST8_MAX, 12u },
            std::array<std::uint_fast8_t, 4u>{ 13u, 14u, 15u, UINT_FAST8_MAX },
          };
          results[j++] = codes[hule.seat()][prev_dapai_seat];
        }
      }
      std::sort(results.begin(), results.end());

      for (std::size_t i = 0u; i < player_annotations.size(); ++i) {
        for (auto const &annotation : player_annotations[i]) {
          if (results[0u] == std::numeric_limits<std::uint_fast8_t>::max()) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
          annotation.printWithRoundResult(
            uuid, i, round_progress, results, round_delta_scores, round_ranks, std::cout);
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
      // 途中流局
      lq::RecordLiuJu record;
      record.ParseFromString(wrapper.data());

      if (record.type() == 1u) {
        // 九種九牌
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
      // 四風連打もしくは四槓散了を確定させる打牌が立直宣言だった場合で，
      // その立直が成立した時，立直の成立を判断するには次の record
      // (つまり lq::RecordNewRound) に記録されている点数
      // (現在の点数より1000点減っているかどうか) を見る以外に方法が無い．
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
          std::array<std::uint_fast8_t, 7u> round_result{
            28u, UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX,
            UINT_FAST8_MAX
          };
          annotation.printWithRoundResult(
            uuid, i, round_progress, round_result, round_delta_scores, round_ranks,
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
      // 開局
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
        std::cout << uuid << ": " << chang << ju << "局" << ben << "本場: "
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
      // 荒牌平局
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
        // 河底牌に対してロンの選択肢が表示されたが見逃した．
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

      std::array<std::uint_fast8_t, 7u> results{
        UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX, UINT_FAST8_MAX,
        UINT_FAST8_MAX, UINT_FAST8_MAX
      };
      {
        std::size_t j = 0u;
        for (std::uint_fast8_t i = 0u; i < 4u; ++i) {
          if (record.players().size() != 4u) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          if (j >= results.size()) {
            KANACHAN_THROW<std::runtime_error>("A broken data.");
          }
          if (record.players()[i].tingpai()) {
            // 聴牌
            results[j++] = 17u + i * 3u;
          }
          else {
            // 不聴
            results[j++] = 16u + i * 3u;
          }

          if (record.liujumanguan()) {
            for (auto const &score : record.scores()) {
              if (score.seat() != i) {
                continue;
              }
              if (j >= results.size()) {
                KANACHAN_THROW<std::runtime_error>("A broken data.");
              }
              results[j++] = 18u + i * 3u;
            }
          }
        }
      }

      for (std::size_t i = 0u; i < player_annotations.size(); ++i) {
        for (auto const &annotation : player_annotations[i]) {
          annotation.printWithRoundResult(
            uuid, i, round_progress, results, round_delta_scores, round_ranks, std::cout);
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
