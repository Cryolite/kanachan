#include "common/throw.hpp"
#include <boost/lexical_cast.hpp>
#include <regex>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <ostream>
#include <ios>
#include <algorithm>
#include <ranges>
#include <iterator>
#include <unordered_map>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <cstddef>


namespace{

namespace views = std::ranges::views;
namespace fs = std::filesystem;

using std::placeholders::_1;

using RoundKey = std::pair<std::string, std::array<std::uint_fast8_t, 3u> >;
struct RoundHash{
  std::size_t operator()(RoundKey const &k) const noexcept
  {
    std::hash<std::string> uuid_hasher;
    std::size_t hash = uuid_hasher(k.first);
    hash ^= (static_cast<std::size_t>(k.second[0u]) << 16u);
    hash ^= (static_cast<std::size_t>(k.second[1u]) <<  8u);
    hash ^= (static_cast<std::size_t>(k.second[2u]) <<  0u);
    std::hash<std::size_t> hasher;
    return hasher(hash);
  }
}; // struct RoundHash
using Paishan = std::vector<std::uint_fast8_t>;
using PaishanMap = std::unordered_map<RoundKey, Paishan, RoundHash>;

template<typename T, typename R>
T castRange(R &&r)
{
  std::stringstream ss;
  for (char const c : r) {
    ss << c;
  }
  T result;
  ss >> result;
  if (ss.fail()) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << std::move(ss).str() << ": Failed to cast.";
  }
  return result;
}

PaishanMap loadPaishanFile(fs::path const &paishan_file_path)
{
  std::ifstream ifs(paishan_file_path);
  std::string line;
  std::regex const e(
    "([[:digit:]]{6}-[[:xdigit:]]{8}(?:-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12})"
    "\t([0-2])"
    "\t([0-3])"
    "\t([[:digit:]]|[1-9][[:digit:]])"
    "\t((?:[[:digit:]]|[12][[:digit:]]|3[0-6])(?:,(?:[[:digit:]]|[12][[:digit:]]|3[0-6])){135})");
  PaishanMap paishan_map;
  for (;;) {
    std::getline(ifs, line);
    if (line.empty()) {
      if (ifs.eof()) {
        break;
      }
      KANACHAN_THROW<std::runtime_error>(_1)
        << paishan_file_path.string() << ": An empty line.";
    }
    if (line.size() == line.max_size()) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << paishan_file_path.string() << ": A too long line.";
    }
    if (ifs.fail() && !ifs.eof() || ifs.bad()) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << paishan_file_path.string() << ": A broken file stream.";
    }

    auto [round_key, paishan_str] = [&]() -> std::pair<RoundKey, std::string>
    {
      std::smatch m;
      if (!std::regex_match(line, m, e)) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << paishan_file_path.string() << ": An invalid line `" << line << "'.";
      }
      std::uint_fast8_t const chang_
        = boost::lexical_cast<unsigned>(m[2u].str());
      std::uint_fast8_t const ju_ = boost::lexical_cast<unsigned>(m[3u].str());
      std::uint_fast8_t const ben_ = boost::lexical_cast<unsigned>(m[4u].str());
      return {
        RoundKey{m[1u].str(), std::array<std::uint_fast8_t, 3u>{chang_, ju_, ben_}},
        m[5u].str()
      };
    }();

    Paishan paishan = [&paishan_str]() -> Paishan
    {
      std::ranges::split_view tile_strs{paishan_str, ','};
      Paishan paishan_;
      paishan_.reserve(136u);
      for (auto tile_str : tile_strs) {
        std::uint_fast8_t const tile = castRange<unsigned>(tile_str);
        if (tile >= 37u) {
          KANACHAN_THROW<std::logic_error>(_1) << tile << ": A logic error.";
        }
        paishan_.push_back(tile);
      }
      return paishan_;
    }();

    {
      auto const [iter, emplaced]
        = paishan_map.try_emplace(std::move(round_key), std::move(paishan));
      if (!emplaced && iter->second != paishan) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << paishan_file_path.string() << ": An wrong file.";
      }
      if (iter->second.size() != 136u) {
        KANACHAN_THROW<std::logic_error>(_1)
          << iter->second.size() << ": A logic error.";
      }
    }
  }

  return paishan_map;
}

using Player = std::tuple<std::uint_fast8_t, std::uint_fast8_t, std::int_fast32_t>;
using Players = std::array<Player, 4u>;
using Sparse = std::vector<std::uint_fast16_t>;
using Numeric = std::tuple<
  std::uint_fast8_t, std::uint_fast8_t,
  std::int_fast32_t, std::int_fast32_t, std::int_fast32_t, std::int_fast32_t>;
using Progression = std::vector<std::uint_fast16_t>;
using Candidates = std::vector<std::uint_fast16_t>;
using Decision = std::tuple<
  Sparse, Numeric, Progression, Candidates, std::uint_fast8_t>;
using Decisions = std::vector<Decision>;
using DeltaScores = std::array<std::int_fast32_t, 4u>;
using Round = std::tuple<Paishan, Decisions, DeltaScores>;
using Rounds = std::vector<Round>;

using GameInfo = std::tuple<std::uint_fast8_t, bool, Players>;
using GameMap = std::unordered_map<std::string, GameInfo>;

using RoundMap = std::unordered_map<RoundKey, Round, RoundHash>;

std::pair<GameMap, RoundMap> parseAnnotations(
  PaishanMap const &paishan_map, fs::path const &annotation_file_path)
{
  std::ifstream ifs(annotation_file_path);
  std::string line;
  std::regex const e(
    "([[:digit:]]{6}-[[:xdigit:]]{8}(?:-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12})"
    "\t([[:digit:]]+(?:,[[:digit:]]+)+)"
    "\t([[:digit:]]+(?:,[[:digit:]]+){5})"
    "\t(0(?:,[[:digit:]]+)*)"
    "\t([[:digit:]]+(?:,[[:digit:]]+)*)"
    "\t([[:digit:]]+)"
    "\t([[:digit:]]+(?:,-?[[:digit:]]+){11})");
  GameMap game_map;
  RoundMap round_map;
  for (;;) {
    std::getline(ifs, line);
    if (line.empty()) {
      if (ifs.eof()) {
        break;
      }
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path.string() << ": An empty line.";
    }
    if (line.size() == line.max_size()) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path.string() << ": A too long line.";
    }
    if (ifs.fail() && !ifs.eof() || ifs.bad()) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path.string() << ": A broken file stream.";
    }

    std::smatch m;
    if (!std::regex_match(line, m, e)) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path.string() << ": An invalid line `" << line << "'.";
    }
    std::string const uuid = m[1u].str();
    std::string const sparse_str = m[2u].str();
    std::string const numeric_str = m[3u].str();
    std::string const progression_str = m[4u].str();
    std::string const candidates_str = m[5u].str();
    std::string const index_str = m[6u].str();
    std::string const result_str = m[7u].str();

    Sparse sparse = [&]() -> Sparse
    {
      Sparse sparse_;
      for (auto feature_str : sparse_str | views::split(',')) {
        std::uint_fast16_t const feature
          = castRange<std::uint_fast16_t>(feature_str);
        if (feature > 525u) {
          KANACHAN_THROW<std::runtime_error>(_1)
            << annotation_file_path.string() << ": " << feature
            << ": An invalid sparse feature.";
        }
        sparse_.push_back(feature);
      }
      return sparse_;
    }();

    std::uint_fast8_t const room = sparse[0u];
    if (room > 4u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path << ": " << sparse[0u] << ": An invalid room.";
    }

    if (sparse[1u] != 5u && sparse[1u] != 6u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path << ": " << sparse[1u]
        << ": An invalid game style.";
    }
    bool const banzhuang_zhan = (sparse[1u] - 5u == 1u);

    std::uint_fast8_t const seat = sparse[2u] - 7u;
    if (seat >= 4u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path.string() << ": " << static_cast<unsigned>(seat)
        << ": An invalid seat.";
    }

    std::uint_fast8_t const chang = sparse[3u] - 11u;
    if (!banzhuang_zhan && chang > 1u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path << ": " << static_cast<unsigned>(chang)
        << ": An invalid chang.";
    }
    if (banzhuang_zhan && chang > 2u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path << ": " << static_cast<unsigned>(chang)
        << ": An invalid chang.";
    }

    std::uint_fast8_t const ju = sparse[4u] - 14u;
    if (ju >= 4u) {
      KANACHAN_THROW<std::runtime_error>(_1)
        << annotation_file_path.string() << ": " << static_cast<unsigned>(ju)
        << ": An invalid ju.";
    }

    Numeric numeric = [&numeric_str]() -> Numeric
    {
      std::ranges::split_view feature_strs{ numeric_str, ',' };
      auto iter = feature_strs.begin();
      std::uint_fast8_t const benchang = castRange<unsigned>(*iter++);
      std::uint_fast8_t const lizhi_deposits = castRange<unsigned>(*iter++);
      std::int_fast32_t const score0 = castRange<std::int_fast32_t>(*iter++);
      std::int_fast32_t const score1 = castRange<std::int_fast32_t>(*iter++);
      std::int_fast32_t const score2 = castRange<std::int_fast32_t>(*iter++);
      std::int_fast32_t const score3 = castRange<std::int_fast32_t>(*iter++);
      if (iter != feature_strs.end()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      return { benchang, lizhi_deposits, score0, score1, score2, score3 };
    }();

    std::uint_fast8_t const ben = std::get<0u>(numeric);

    RoundKey round_key{uuid, std::array<std::uint_fast8_t, 3u>{chang, ju, ben}};

    Paishan const &paishan = [&paishan_map, &round_key]() -> Paishan const &
    {
      PaishanMap::const_iterator const found = paishan_map.find(round_key);
      if (found == paishan_map.cend()) {
        std::string const &uuid = std::get<0u>(round_key);
        unsigned const chang = std::get<1u>(round_key)[0u];
        unsigned const ju = std::get<1u>(round_key)[1u];
        unsigned const ben = std::get<1u>(round_key)[2u];
        KANACHAN_THROW<std::runtime_error>(_1)
          << uuid << ',' << chang << ',' << ju << ',' << ben
          << ": No pai shan found.";
      }
      return found->second;
    }();

    Progression progression = [&]() -> Progression
    {
      Progression progression_;
      for (auto feature_str : progression_str | views::split(',')) {
        std::uint_fast16_t const feature
          = castRange<std::uint_fast16_t>(feature_str);
        if (feature > 2164u) {
          KANACHAN_THROW<std::runtime_error>(_1)
            << annotation_file_path.string() << ": " << feature
            << ": An invalid progression feature.";
        }
        progression_.push_back(feature);
      }
      return progression_;
    }();

    Candidates candidates = [&]() -> Candidates
    {
      Candidates candidates_;
      for (auto feature_str : candidates_str | views::split(',')) {
        std::uint_fast16_t const feature
          = castRange<std::uint_fast16_t>(feature_str);
        if (feature > 545u) {
          KANACHAN_THROW<std::runtime_error>(_1)
            << annotation_file_path.string() << ": " << feature
            << ": An invalid candidate.";
        }
        candidates_.push_back(feature);
      }
      return candidates_;
    }();

    std::uint_fast8_t const index = boost::lexical_cast<unsigned>(index_str);
    if (index >= candidates.size()) {
      KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
    }

    auto const [delta_scores, final_ranking, final_score]
      = [&]() -> std::tuple<DeltaScores, std::uint_fast8_t, std::int_fast32_t>
    {
      std::ranges::split_view feature_strs{result_str, ','};
      auto iter = feature_strs.begin();
      ++iter;
      DeltaScores delta_scores_{
        castRange<std::int_fast32_t>(*iter++),
        castRange<std::int_fast32_t>(*iter++),
        castRange<std::int_fast32_t>(*iter++),
        castRange<std::int_fast32_t>(*iter++)
      };
      std::rotate(
        delta_scores_.begin(), delta_scores_.end() - seat, delta_scores_.end());
      ++iter;
      ++iter;
      ++iter;
      ++iter;
      std::uint_fast8_t const final_ranking_ = castRange<unsigned>(*iter++);
      if (final_ranking_ >= 4u) {
        KANACHAN_THROW<std::runtime_error>(_1)
          << annotation_file_path << ": "
          << static_cast<unsigned>(final_ranking_)
          << ": An invalid final ranking.";
      }
      std::int_fast32_t const final_score_
        = castRange<std::int_fast32_t>(*iter++);
      ++iter;
      if (iter != feature_strs.end()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      return {delta_scores_, final_ranking_, final_score_};
    }();

    if (chang == 0u && ju == 0u && ben == 0u && progression.size() == 1u) {
      if (progression.front() != 0u) {
          KANACHAN_THROW<std::runtime_error>(_1)
            << annotation_file_path.string() << ": An wrong annotation.";
      }
      Players players{
        Player{
          static_cast<std::uint_fast8_t>(-1),
          static_cast<std::uint_fast8_t>(-1),
          std::numeric_limits<std::int_fast32_t>::max()
        },
        Player{
          static_cast<std::uint_fast8_t>(-1),
          static_cast<std::uint_fast8_t>(-1),
          std::numeric_limits<std::int_fast32_t>::max()
        },
        Player{
          static_cast<std::uint_fast8_t>(-1),
          static_cast<std::uint_fast8_t>(-1),
          std::numeric_limits<std::int_fast32_t>::max()
        },
        Player{
          static_cast<std::uint_fast8_t>(-1),
          static_cast<std::uint_fast8_t>(-1),
          std::numeric_limits<std::int_fast32_t>::max()
        }
      };
      for (std::uint_fast16_t const f : sparse) {
        if (273u <= f && f <= 288u) {
          std::get<0u>(players[0u]) = f - 273u;
        }
        if (293u <= f && f <= 308u) {
          std::get<0u>(players[1u]) = f - 293u;
        }
        if (313u <= f && f <= 328u) {
          std::get<0u>(players[2u]) = f - 313u;
        }
        if (333u <= f && f <= 348u) {
          std::get<0u>(players[3u]) = f - 333u;
        }
      }
      auto const [found, emplaced]
        = game_map.try_emplace(uuid, room, banzhuang_zhan, players);
      if (!emplaced) {
        if (room != std::get<0u>(found->second)) {
          KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
        }
        if (banzhuang_zhan != std::get<1u>(found->second)) {
          KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
        }
        if (std::get<0u>(players[0u]) != std::get<0u>(std::get<2u>(found->second)[0u])) {
          KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
        }
        if (std::get<0u>(players[1u]) != std::get<0u>(std::get<2u>(found->second)[1u])) {
          KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
        }
        if (std::get<0u>(players[2u]) != std::get<0u>(std::get<2u>(found->second)[2u])) {
          KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
        }
        if (std::get<0u>(players[3u]) != std::get<0u>(std::get<2u>(found->second)[3u])) {
          KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
        }
      }
    }
    {
      auto const found = game_map.find(uuid);
      if (found == game_map.cend()) {
        KANACHAN_THROW<std::logic_error>("A logic error.");
      }
      Player &player = std::get<2u>(found->second)[seat];
      std::get<1u>(player) = final_ranking;
      std::get<2u>(player) = final_score;
    }

    {
      auto const [found, emplaced]
        = round_map.try_emplace(round_key, paishan, Decisions{}, delta_scores);
      if (!emplaced) {
        if (paishan != std::get<0u>(found->second)) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
        if (delta_scores != std::get<2u>(found->second)) {
          KANACHAN_THROW<std::runtime_error>(_1)
            << annotation_file_path.string() << ": An wrong annotation.";
        }
      }
      {
        Decision decision{
          std::move(sparse), numeric, std::move(progression),
          std::move(candidates), index
        };
        std::get<1u>(found->second).push_back(std::move(decision));
      }
    }
  }

  return {std::move(game_map), std::move(round_map)};
}

struct DecisionLess
{
private:
  static std::uint_fast8_t getNumLeftTiles(Sparse const &sparse)
  {
    for (std::uint_fast16_t const f : sparse) {
      if (203u <= f && f <= 272u) {
        return f - 203u;
      }
    }
    KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
  }

public:
  bool operator()(Decision const &lhs, Decision const &rhs) const
  {
    Sparse const &sparse0 = std::get<0u>(lhs);
    Numeric const &numeric0 = std::get<1u>(lhs);
    Progression const &progression0 = std::get<2u>(lhs);
    std::uint_fast8_t const chang0 = sparse0[3u] - 11u;
    std::uint_fast8_t const ju0 = sparse0[4u] - 14u;
    std::uint_fast8_t const ben0 = std::get<0u>(numeric0);
    std::uint_fast8_t const turn0 = progression0.size();
    std::uint_fast8_t const num_left_tiles0 = getNumLeftTiles(sparse0);

    Sparse const &sparse1 = std::get<0u>(rhs);
    Numeric const &numeric1 = std::get<1u>(rhs);
    Progression const &progression1 = std::get<2u>(rhs);
    std::uint_fast8_t const chang1 = sparse1[3u] - 11u;
    std::uint_fast8_t const ju1 = sparse1[4u] - 14u;
    std::uint_fast8_t const ben1 = std::get<0u>(numeric1);
    std::uint_fast8_t const turn1 = progression1.size();
    std::uint_fast8_t const num_left_tiles1 = getNumLeftTiles(sparse1);

    if (chang0 < chang1) {
      return true;
    }
    if (chang0 > chang1) {
      return false;
    }

    if (ju0 < ju1) {
      return true;
    }
    if (ju0 > ju1) {
      return false;
    }

    if (ben0 < ben1) {
      return true;
    }
    if (ben0 > ben1) {
      return false;
    }

    if (turn0 < turn1) {
      return true;
    }
    if (turn0 > turn1) {
      return false;
    }

    if (num_left_tiles0 > num_left_tiles1) {
      return true;
    }
    if (num_left_tiles0 < num_left_tiles1) {
      return false;
    }

    if (progression0 != progression1) {
      KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
    }

    if (sparse0 == sparse1) {
      if (numeric0 != numeric1) {
        KANACHAN_THROW<std::runtime_error>("An wrong annotation.");
      }
      return false;
    }

    // 同じ打牌に対する各席の decision を打牌した席からの席順で並べる．
    std::uint_fast8_t const dapai_seat = [&progression0]() -> std::uint_fast8_t
    {
      std::uint_fast16_t const encode = progression0.back();
      if (5u <= encode && encode <= 596u) {
        // 通常の打牌．
        return (encode - 5u) / 148u;
      }
      if (1881u <= encode && encode <= 2016u) {
        // 槍槓があるので暗槓を打牌とみなす．
        return (encode - 1881u) / 34u;
      }
      if (2017u <= encode && encode <= 2164u) {
        // 槍槓があるので加槓を打牌とみなす．
        return (encode - 2017u) / 37u;
      }
      KANACHAN_THROW<std::runtime_error>(_1)
        << encode << ": An wrong annotation";
    }();
    std::uint_fast8_t const seat0 = sparse0[2u] - 7u;
    std::uint_fast8_t const relseat0 = (seat0 + 4u - dapai_seat) % 4u;
    std::uint_fast8_t const seat1 = sparse1[2u] - 7u;
    std::uint_fast8_t const relseat1 = (seat1 + 4u - dapai_seat) % 4u;
    if (relseat0 < relseat1) {
      return true;
    }
    if (relseat0 > relseat1) {
      return false;
    }

    return false;
  }
}; // struct DecisionLess

void sortUniqueDecisions(Decisions &decisions)
{
  std::ranges::sort(decisions, DecisionLess{});
  auto const r = std::ranges::unique(decisions);
  decisions.erase(r.begin(), decisions.end());
}

void printPlayer(Player const &player, std::ostream &os)
{
  os << "{\"grade\":" << static_cast<unsigned>(std::get<0u>(player))
     << ",\"final_ranking\":" << static_cast<unsigned>(std::get<1u>(player))
     << ",\"final_score\":" << std::get<2u>(player) << '}';
}

void printPlayers(Players const &players, std::ostream &os)
{
  os << '[';
  bool flag = false;
  for (Player const &player : players) {
    if (flag) {
      os << ',';
    }
    printPlayer(player, os);
    flag = true;
  }
  os << ']';
}

void printPaishan(Paishan const &paishan, std::ostream &os)
{
  os << '[';
  bool flag = false;
  for (std::uint_fast8_t const tile : paishan) {
    if (flag) {
      os << ',';
    }
    os << static_cast<unsigned>(tile);
    flag = true;
  }
  os << ']';
}

void printVector(std::vector<std::uint_fast16_t> const &v, std::ostream &os)
{
  os << '[';
  bool flag = false;
  for (std::uint_fast16_t const e : v) {
    if (flag) {
      os << ',';
    }
    os << e;
    flag = true;
  }
  os << ']';
}

void printNumeric(Numeric const &numeric, std::ostream &os)
{
  os << '[' << static_cast<unsigned>(std::get<0u>(numeric))
     << ',' << static_cast<unsigned>(std::get<1u>(numeric))
     << ',' << std::get<2u>(numeric) << ',' << std::get<3u>(numeric)
     << ',' << std::get<4u>(numeric) << ',' << std::get<5u>(numeric) << ']';
}

void printDecision(Decision const &decision, std::ostream &os)
{
  auto const &[sparse, numeric, progression, candidates, index] = decision;
  os << "{\"sparse\":";
  printVector(sparse, os);
  os << ",\"numeric\":";
  printNumeric(numeric, os);
  os << ",\"progression\":";
  printVector(progression, os);
  os << ",\"candidates\":";
  printVector(candidates, os);
  os << ",\"index\":" << static_cast<unsigned>(index) << '}';
}

void printDecisions(Decisions const &decisions, std::ostream &os)
{
  os << '[';
  bool flag = false;
  for (Decision const &decision : decisions) {
    if (flag) {
      os << ',';
    }
    printDecision(decision, os);
    flag = true;
  }
  os << ']';
}

void printDeltaScores(DeltaScores const &delta_scores, std::ostream &os)
{
  os << '[';
  bool flag = false;
  for (std::int_fast32_t const delta_score : delta_scores) {
    if (flag) {
      os << ',';
    }
    os << delta_score;
    flag = true;
  }
  os << ']';
}

void printRound(Round const &round, std::ostream &os)
{
  auto const &[paishan, decisions, delta_scores] = round;
  os << "{\"paishan\":";
  printPaishan(paishan, os);
  os << ",\"decisions\":";
  printDecisions(decisions, os);
  os << ",\"delta_scores\":";
  printDeltaScores(delta_scores, os);
  os << '}';
}

void printRounds(Rounds const &rounds, std::ostream &os)
{
  os << '[';
  bool flag = false;
  for (Round const &round : rounds) {
    if (flag) {
      os << ',';
    }
    printRound(round, os);
    flag = true;
  }
  os << ']';
}

using RoundsMap = std::unordered_map<std::string, Rounds>;

struct RoundLess
{
  bool operator()(Round const &lhs, Round const &rhs) const
  {
    auto const &[paishan0, decisions0, delta_scores0] = lhs;
    if (decisions0.empty()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    Decision const &decision0 = decisions0.front();
    Sparse const &sparse0 = std::get<0u>(decision0);
    Numeric const &numeric0 = std::get<1u>(decision0);
    std::uint_fast8_t const chang0 = sparse0[3u] - 11u;
    std::uint_fast8_t const ju0 = sparse0[4u] - 14u;
    std::uint_fast8_t const ben0 = std::get<0u>(numeric0);

    auto const &[paishan1, decisions1, delta_scores1] = rhs;
    if (decisions1.empty()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    Decision const &decision1 = decisions1.front();
    Sparse const &sparse1 = std::get<0u>(decision1);
    Numeric const &numeric1 = std::get<1u>(decision1);
    std::uint_fast8_t const chang1 = sparse1[3u] - 11u;
    std::uint_fast8_t const ju1 = sparse1[4u] - 14u;
    std::uint_fast8_t const ben1 = std::get<0u>(numeric1);

    if (chang0 < chang1) {
      return true;
    }
    if (chang0 > chang1) {
      return false;
    }

    if (ju0 < ju1) {
      return true;
    }
    if (ju0 > ju1) {
      return false;
    }

    if (ben0 < ben1) {
      return true;
    }
    if (ben0 > ben1) {
      return false;
    }

    return false;
  }
}; // struct RoundLess

} // namespace *unnamed*

int main(int const argc, char const * const * const argv)
{
  if (argc == 0) {
    KANACHAN_THROW<std::runtime_error>(_1) << "argc == " << argc;
  }
  if (argc < 4) {
    KANACHAN_THROW<std::runtime_error>("Too few arguments.");
  }
  if (argc > 4) {
    KANACHAN_THROW<std::runtime_error>("Too many arguments.");
  }

  fs::path paishan_file_path(argv[1u]);
  if (!fs::exists(paishan_file_path)) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << paishan_file_path.string() << ": Does not exist.";
  }
  if (fs::is_directory(paishan_file_path)) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << paishan_file_path.string() << ": Not a file.";
  }

  fs::path annotation_file_path(argv[2u]);
  if (!fs::exists(annotation_file_path)) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << annotation_file_path.string() << ": Does not exist.";
  }
  if (fs::is_directory(annotation_file_path)) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << annotation_file_path.string() << ": Not a file";
  }

  fs::path output_prefix(argv[3u]);
  if (fs::exists(output_prefix) && !fs::is_directory(output_prefix)) {
    KANACHAN_THROW<std::runtime_error>(_1)
      << output_prefix.string() << ": Not a directory.";
  }

  PaishanMap paishan_map = loadPaishanFile(paishan_file_path);
  auto [game_map, round_map]
    = parseAnnotations(paishan_map, annotation_file_path);

  RoundsMap rounds_map;
  for (auto &[round_key, round] : round_map) {
    std::string const &uuid = std::get<0u>(round_key);
    Decisions &decisions = std::get<1u>(round);
    sortUniqueDecisions(decisions);
    RoundsMap::iterator found = rounds_map.find(uuid);
    if (found == rounds_map.end()) {
      std::tie(found, std::ignore) = rounds_map.try_emplace(uuid, Rounds{});
    }
    found->second.push_back(std::move(round));
  }

  for (auto &[uuid, rounds] : rounds_map) {
    std::ranges::sort(rounds, RoundLess{});
  }

  for (auto const &[uuid, game_info] : game_map) {
    auto const &[room, banzhuang_zhan, players] = game_info;
    for (Player const &player : players) {
      if (std::get<1u>(player) == static_cast<std::uint_fast8_t>(-1)) {
        KANACHAN_THROW<std::runtime_error>(
          "A player does not have any decision.");
      }
      if (std::get<2u>(player) == std::numeric_limits<std::int_fast32_t>::max()) {
        KANACHAN_THROW<std::runtime_error>(
          "A player does not have any decision.");
      }
    }

    RoundsMap::const_iterator found = std::as_const(rounds_map).find(uuid);
    if (found == rounds_map.cend()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    Rounds const &rounds = found->second;

    fs::path const output_file_path = output_prefix / (uuid + ".json");
    fs::create_directories(output_prefix);
    std::ofstream ofs(output_file_path, std::ios_base::out);
    ofs << '{' << "\"uuid\":\"" << uuid << "\",\"room\":" << static_cast<unsigned>(room)
        << ",\"style\":" << (banzhuang_zhan ? 1u : 0u) << ",\"players\":";
    printPlayers(players, ofs);
    ofs << ",\"rounds\":";
    printRounds(rounds, ofs);
    ofs << '}' << std::flush;
  }

  return EXIT_SUCCESS;
}
