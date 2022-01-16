#include "common/throw.hpp"
#include "common/mahjongsoul.pb.h"
#include <regex>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ios>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <functional>
#include <utility>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <cstddef>


namespace {

using std::placeholders::_1;

std::uint_fast8_t decodeTile(char const number, char const color)
{
  if (number < '0' || '9' < number) {
    KANACHAN_THROW<std::runtime_error>(_1) << number;
  }
  if (color != 'm' && color != 'p' && color != 's' && color != 'z') {
    KANACHAN_THROW<std::runtime_error>(_1) << color;
  }

  std::uint_fast8_t const number_ = number - '0';
  std::uint_fast8_t const color_ = [](char const c) -> std::uint_fast8_t {
    switch (c) {
    case 'm':
      return 0u;
    case 'p':
      return 10u;
    case 's':
      return 20u;
    case 'z':
      return 29u;
    default:
      KANACHAN_THROW<std::logic_error>("");
    }
  }(color);

  std::uint_fast8_t const encode = color_ + number_;
  if (encode >= 37u) {
    KANACHAN_THROW<std::runtime_error>(_1) << encode;
  }

  return encode;
}

std::string restorePaishan(
  std::vector<std::string> &qipai0,
  std::vector<std::string> &qipai1,
  std::vector<std::string> &qipai2,
  std::vector<std::string> &qipai3,
  std::string const &left_tiles,
  std::mt19937 &urng)
{
  if (qipai0.size() != 14u) {
    KANACHAN_THROW<std::runtime_error>(_1) << qipai0.size();
  }
  if (qipai1.size() != 13u) {
    KANACHAN_THROW<std::runtime_error>(_1) << qipai1.size();
  }
  if (qipai2.size() != 13u) {
    KANACHAN_THROW<std::runtime_error>(_1) << qipai2.size();
  }
  if (qipai3.size() != 13u) {
    KANACHAN_THROW<std::runtime_error>(_1) << qipai3.size();
  }
  if (left_tiles.size() != 83u * 2u) {
    KANACHAN_THROW<std::runtime_error>(_1) << left_tiles.size();
  }

  auto encode = [](std::string const &tile) -> std::uint_fast8_t {
    if (tile.size() != 2u) {
      KANACHAN_THROW<std::runtime_error>(_1) << tile.size();
    }
    return decodeTile(tile[0u], tile[1u]);
  };

  std::array<std::uint_fast8_t, 14u> qipai0_;
  std::transform(qipai0.cbegin(), qipai0.cend(), qipai0_.begin(), encode);
  std::array<std::uint_fast8_t, 13u> qipai1_;
  std::transform(qipai1.cbegin(), qipai1.cend(), qipai1_.begin(), encode);
  std::array<std::uint_fast8_t, 13u> qipai2_;
  std::transform(qipai2.cbegin(), qipai2.cend(), qipai2_.begin(), encode);
  std::array<std::uint_fast8_t, 13u> qipai3_;
  std::transform(qipai3.cbegin(), qipai3.cend(), qipai3_.begin(), encode);

  // 番号が最も大きい牌を親の第一自摸牌にしなければならない．
  std::sort(qipai0_.begin(), qipai0_.end());
  // 数字による単純な sort では赤牌が後ろに来ないので修正が必要となる．
  if (qipai0_.back() <= 5u) {
    auto const found0 = std::find(qipai0_.begin(), qipai0_.end(), 0u);
    if (found0 != qipai0_.cend()) {
      std::copy(found0 + 1, qipai0_.end(), found0);
      auto const found1 = std::find(qipai0_.begin(), qipai0_.end() - 1, 5u);
      std::copy_backward(found1, qipai0_.end() - 1, qipai0_.end());
      *found1 = 0u;
    }
  }
  if (qipai0_.back() <= 15u) {
    auto const found0 = std::find(qipai0_.begin(), qipai0_.end(), 10u);
    if (found0 != qipai0_.cend()) {
      std::copy(found0 + 1, qipai0_.end(), found0);
      auto const found1 = std::find(qipai0_.begin(), qipai0_.end() - 1, 15u);
      std::copy_backward(found1, qipai0_.end() - 1, qipai0_.end());
      *found1 = 10u;
    }
  }
  if (qipai0_.back() <= 25u) {
    auto const found0 = std::find(qipai0_.begin(), qipai0_.end(), 20u);
    if (found0 != qipai0_.cend()) {
      std::copy(found0 + 1, qipai0_.end(), found0);
      auto const found1 = std::find(qipai0_.begin(), qipai0_.end() - 1, 25u);
      std::copy_backward(found1, qipai0_.end() - 1, qipai0_.end());
      *found1 = 20u;
    }
  }

  std::shuffle(qipai0_.begin(), qipai0_.end() - 1, urng);
  std::shuffle(qipai1_.begin(), qipai1_.end(), urng);
  std::shuffle(qipai2_.begin(), qipai2_.end(), urng);
  std::shuffle(qipai3_.begin(), qipai3_.end(), urng);

  std::ostringstream oss;

  auto print = [&oss] (std::uint_fast8_t const tile, bool const end) -> void {
    oss << static_cast<unsigned>(tile);
    if (!end) {
      oss << ',';
    }
  };

  for (std::uint_fast8_t i = 0u; i < 12u; i += 4u) {
    for (std::uint_fast8_t j = i; j < i + 4u; ++j) {
      print(qipai0_[j], false);
    }
    for (std::uint_fast8_t j = i; j < i + 4u; ++j) {
      print(qipai1_[j], false);
    }
    for (std::uint_fast8_t j = i; j < i + 4u; ++j) {
      print(qipai2_[j], false);
    }
    for (std::uint_fast8_t j = i; j < i + 4u; ++j) {
      print(qipai3_[j], false);
    }
  }
  print(qipai0_[12u], false);
  print(qipai1_[12u], false);
  print(qipai2_[12u], false);
  print(qipai3_[12u], false);
  print(qipai0_[13u], true);

  for (std::string::const_iterator iter = left_tiles.cbegin(); iter < left_tiles.cend(); iter += 2) {
    oss << ',' << static_cast<unsigned>(decodeTile(*iter, *(iter + 1)));
  }

  return std::move(oss).str();
}

void process(std::filesystem::path const &ph, std::mt19937 &urng) {
  std::string data = [&ph]() {
    std::ifstream ifs(ph, std::ios_base::in | std::ios_base::binary);
    for (std::size_t i = 0; i < 3u; ++i) {
      std::ifstream::int_type const c = ifs.get();
      if (c == std::ifstream::traits_type::eof()) {
        throw std::runtime_error("");
      }
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return std::move(oss).str();
  }();

  lq::Wrapper wrapper;
  wrapper.ParseFromString(data);
  if (wrapper.name() != "") {
    throw std::runtime_error("");
  }

  lq::ResGameRecord msg0;
  msg0.ParseFromString(wrapper.data());

  std::string const uuid = msg0.head().uuid();

  wrapper.ParseFromString(msg0.data());
  if (wrapper.name() != ".lq.GameDetailRecords") {
    throw std::runtime_error("");
  }

  lq::GameDetailRecords msg1;
  msg1.ParseFromString(wrapper.data());

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

  std::string chang;
  std::size_t ju;
  std::size_t ben;

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
      lq::RecordAnGangAddGang record;
      record.ParseFromString(wrapper.data());
    }
    else if (wrapper.name() == ".lq.RecordChiPengGang") {
      lq::RecordChiPengGang record;
      record.ParseFromString(wrapper.data());
    }
    else if (wrapper.name() == ".lq.RecordDealTile") {
      lq::RecordDealTile record;
      record.ParseFromString(wrapper.data());
    }
    else if (wrapper.name() == ".lq.RecordDiscardTile") {
      lq::RecordDiscardTile record;
      record.ParseFromString(wrapper.data());
    }
    else if (wrapper.name() == ".lq.RecordHule") {
      lq::RecordHule record;
      record.ParseFromString(wrapper.data());
    }
    else if (wrapper.name() == ".lq.RecordLiuJu") {
      lq::RecordLiuJu record;
      record.ParseFromString(wrapper.data());
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

      std::vector<std::string> qipai0;
      std::vector<std::string> qipai1;
      std::vector<std::string> qipai2;
      std::vector<std::string> qipai3;
      switch (record.ju()) {
      case 0:
        qipai0.assign(record.tiles0().cbegin(), record.tiles0().cend());
        qipai1.assign(record.tiles1().cbegin(), record.tiles1().cend());
        qipai2.assign(record.tiles2().cbegin(), record.tiles2().cend());
        qipai3.assign(record.tiles3().cbegin(), record.tiles3().cend());
        break;
      case 1:
        qipai0.assign(record.tiles1().cbegin(), record.tiles1().cend());
        qipai1.assign(record.tiles2().cbegin(), record.tiles2().cend());
        qipai2.assign(record.tiles3().cbegin(), record.tiles3().cend());
        qipai3.assign(record.tiles0().cbegin(), record.tiles0().cend());
        break;
      case 2:
        qipai0.assign(record.tiles2().cbegin(), record.tiles2().cend());
        qipai1.assign(record.tiles3().cbegin(), record.tiles3().cend());
        qipai2.assign(record.tiles0().cbegin(), record.tiles0().cend());
        qipai3.assign(record.tiles1().cbegin(), record.tiles1().cend());
        break;
      case 3:
        qipai0.assign(record.tiles3().cbegin(), record.tiles3().cend());
        qipai1.assign(record.tiles0().cbegin(), record.tiles0().cend());
        qipai2.assign(record.tiles1().cbegin(), record.tiles1().cend());
        qipai3.assign(record.tiles2().cbegin(), record.tiles2().cend());
        break;
      default:
        KANACHAN_THROW<std::runtime_error>(_1) << record.ju();
      }

      std::string paishan = restorePaishan(
        qipai0, qipai1, qipai2, qipai3, record.paishan(), urng);

      std::cout << uuid << '\t' << record.chang() << '\t' << record.ju()
                << '\t' << record.ben() << '\t' << paishan << '\n';
    }
    else if (wrapper.name() == ".lq.RecordNoTile") {
      lq::RecordNoTile record;
      record.ParseFromString(wrapper.data());
    }
    else {
      KANACHAN_THROW<std::runtime_error>(wrapper.name());
    }
  }
}

void walk(std::filesystem::path const &ph, std::mt19937 &urng) {
  auto const end = std::filesystem::directory_iterator();
  for (std::filesystem::directory_iterator iter(ph); iter != end; ++iter) {
    if (iter->is_directory()) {
      walk(iter->path(), urng);
      continue;
    }
    if (iter->is_regular_file()) {
      process(iter->path(), urng);
      continue;
    }
  }
}

std::vector<std::uint_least32_t> getRandomSeed()
{
  constexpr std::size_t state_size = std::mt19937::state_size;

  std::random_device rand;
  std::vector<std::uint_least32_t> seed;
  seed.reserve(state_size);
  for (std::size_t i = 0; i < state_size; ++i) {
    std::uint_least32_t const seed_ = rand();
    seed.push_back(seed_);
  }
  return seed;
}

} // namespace *unnamed*

int main(int const argc, char const *argv[]) {
  if (argc < 2) {
    std::exit(EXIT_FAILURE);
  }

  std::vector<std::uint_least32_t> seed = getRandomSeed();
  std::seed_seq ss(seed.cbegin(), seed.cend());
  std::mt19937 urng(ss);

  std::filesystem::path ph(argv[1]);
  walk(ph, urng);

  return EXIT_SUCCESS;
}
