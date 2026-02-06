// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_GOMUKO_H_
#define OPEN_SPIEL_GAMES_GOMUKO_H_

#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace gomuko {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 7;
inline constexpr int kNumCols = 7;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kWinSize = 4;
inline constexpr int kCellStates = 1 + kNumPlayers; // empty, 'x', and 'o'.

// https://math.stackexchange.com/questions/485752/Gomuko-state-space-choose-calculation/485852
// inline constexpr int kNumberStates = 5478; No Idea how to calculate this,
// back later if this constants is needed

// State of a cell.
enum class CellState {
  kEmpty,
  kNought, // O
  kCross,  // X
};

struct GomukoStructContents {
  std::string current_player;
  std::vector<std::string> board;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(GomukoStructContents, current_player, board);
};

struct GomukoStateStruct : public StateStruct, public GomukoStructContents {
  GomukoStateStruct() = default;
  explicit GomukoStateStruct(const std::string &json_str) {
    nlohmann::json::parse(json_str).get_to(*this);
  }

  nlohmann::json to_json_base() const override { return *this; }
};

struct GomukoObservationStruct : public ObservationStruct,
                                 public GomukoStructContents {
  GomukoObservationStruct() = default;
  explicit GomukoObservationStruct(const std::string &json_str) {
    nlohmann::json::parse(json_str).get_to(*this);
  }

  nlohmann::json to_json_base() const override { return *this; }
};

struct GomukoActionStruct : public ActionStruct {
  int row;
  int col;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(GomukoActionStruct, row, col);

  GomukoActionStruct() = default;
  explicit GomukoActionStruct(const std::string &json_str) {
    nlohmann::json::parse(json_str).get_to(*this);
  }

  nlohmann::json to_json_base() const override { return *this; }
};

// State of an in-play game.
class GomukoState : public State {
public:
  GomukoState(std::shared_ptr<const Game> game);
  GomukoState(std::shared_ptr<const Game> game, const nlohmann::json &json);

  GomukoState(const GomukoState &) = default;
  GomukoState &operator=(const GomukoState &) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  std::vector<CellState> Board() const;
  CellState BoardAt(int cell) const { return board_[cell]; }
  CellState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }
  Player outcome() const { return outcome_; }
  void ChangePlayer() { current_player_ = current_player_ == 0 ? 1 : 0; }

  void SetCurrentPlayer(Player player) { current_player_ = player; }

  std::unique_ptr<StateStruct> ToStruct() const override;
  std::unique_ptr<ObservationStruct>
  ToObservationStruct(Player player) const override;
  std::unique_ptr<ActionStruct> ActionToStruct(Player player,
                                               Action action_id) const override;
  Action StructToAction(const ActionStruct &action_struct) const override;

protected:
  std::array<CellState, kNumCells> board_;
  void DoApplyAction(Action move) override;

private:
  bool
  HasLine(Player player,
          Action action) const; // Does this player have a line at this action?
  bool IsFull() const;          // Is the board full?
  Player current_player_ = 0;   // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
};

// Game object.
class GomukoGame : public Game {
public:
  explicit GomukoGame(const GameParameters &params);
  int NumDistinctActions() const override { return kNumCells; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new GomukoState(shared_from_this()));
  }
  // std::unique_ptr<State> NewInitialState(const nlohmann::json &json) const {
  //   return std::unique_ptr<State>(new GomukoState(shared_from_this(), json));
  // }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return kNumCells; }
  std::string ActionToString(Player player, Action action_id) const override;
};

CellState PlayerToState(Player player);
std::string PlayerToString(Player player);
std::string StateToString(CellState state);

// Does this player have a line at this action?
bool BoardHasLine(const std::array<CellState, kNumCells> &board,
                  const Player player, const Action action);

inline std::ostream &operator<<(std::ostream &stream, const CellState &state) {
  return stream << StateToString(state);
}

} // namespace gomuko
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_GOMUKO_H_
