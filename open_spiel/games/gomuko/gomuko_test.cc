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

#include <iostream>
#include <string>
#include <vector>

#include "open_spiel/games/gomuko/gomuko.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace gomuko {
namespace {

namespace testing = open_spiel::testing;

struct ActionStruct {
  Action action;
  Player player;
};

void BasicGomukoTests() {
  testing::LoadGameTest("gomuko");
  testing::NoChanceOutcomesTest(*LoadGame("gomuko"));
  testing::RandomSimTest(*LoadGame("gomuko"), 100);
}

std::string CreateEmptyBoard(int size) {
  std::string board_state = "{\"board\":[";

  for (int i = 0; i < size; ++i) {
    board_state += "\".\"";
    if (i < size - 1) {
      board_state += ",";
    }
  }

  board_state += "],";

  return board_state + "\"current_player\":\"x\"}";
}

std::string CreateBoardWithListAction(int size,
                                      std::vector<ActionStruct> actions) {
  std::string board_state = "{\"board\":[";

  std::sort(actions.begin(), actions.end(),
            [](const ActionStruct &a, const ActionStruct &b) {
              return a.action < b.action;
            });

  int action_index = 0;

  for (int i = 0; i < size; ++i) {
    std::string cur = StateToString(CellState::kEmpty);

    if (action_index < actions.size() && i == actions[action_index].action) {
      cur = PlayerToString(actions[action_index].player);
      action_index += 1;
    }

    board_state += "\"" + cur + "\"";

    if (i < size - 1) {
      board_state += ",";
    }
  }
  board_state += "],";

  std::string current_player;

  if (actions.size() % 2 != 0) {
    current_player = PlayerToString(1);
  } else {
    current_player = PlayerToString(0);
  }

  return board_state + "\"current_player\":\"" + current_player + "\"}";
}

void TestStateStruct() {
  auto game = LoadGame("gomuko");
  auto state = game->NewInitialState();
  GomukoState *ttt_state = static_cast<GomukoState *>(state.get());
  auto state_struct = ttt_state->ToStruct();
  // Test state/state_struct -> json string.
  SPIEL_CHECK_EQ(state_struct->ToJson(), ttt_state->ToJson());

  std::string state_json = CreateEmptyBoard(7 * 7);
  SPIEL_CHECK_EQ(state_struct->ToJson(), state_json);
  // Test json string -> state_struct.
  SPIEL_CHECK_EQ(nlohmann::json::parse(state_json).dump(),
                 GomukoStateStruct(state_json).ToJson());
}

void TestObservationStruct() {
  auto game = LoadGame("gomuko");
  auto state = game->NewInitialState();
  state->ApplyAction(4);
  GomukoState *ttt_state = static_cast<GomukoState *>(state.get());
  auto obs_struct = ttt_state->ToObservationStruct(0);

  std::vector<ActionStruct> actions = {};
  actions.push_back({4, 0});
  std::string obs_json = CreateBoardWithListAction(7 * 7, actions);

  SPIEL_CHECK_EQ(obs_struct->ToJson(), obs_json);
  SPIEL_CHECK_EQ(nlohmann::json::parse(obs_json).dump(),
                 GomukoObservationStruct(obs_json).ToJson());
}

void TestActionStruct() {
  auto game = LoadGame("gomuko");
  auto state = game->NewInitialState();
  auto *ttt_state = static_cast<GomukoState *>(state.get());

  // Test ActionToStruct.
  Action action_id = 24; // Player 0 plays in the center.
  auto action_struct = ttt_state->ActionToStruct(0, action_id);
  std::string action_json = "{\"col\":3,\"row\":3}";
  SPIEL_CHECK_EQ(action_struct->ToJson(), action_json);

  // Test ApplyActionStruct.
  auto state2 = game->NewInitialState();
  state2->ApplyActionStruct(*action_struct);
  SPIEL_CHECK_EQ(
      state2->ToString(),
      ".......\n.......\n.......\n...x...\n.......\n.......\n.......");

  // Test JSON parsing.
  SPIEL_CHECK_EQ(nlohmann::json::parse(action_json).dump(),
                 GomukoActionStruct(action_json).ToJson());

  // Test StructToAction.
  SPIEL_CHECK_EQ(action_id, ttt_state->StructToAction(*action_struct));
}

void Player1Win() {
  auto game = LoadGame("gomuko");
  auto state = game->NewInitialState();
  state->ApplyAction(0);
  state->ApplyAction(8);
  state->ApplyAction(1);
  state->ApplyAction(9);
  state->ApplyAction(2);
  state->ApplyAction(10);
  state->ApplyAction(3);

  std::cout << state->ToString();

  SPIEL_CHECK_EQ(state->IsTerminal(), true);
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

void Player2Win() {
  auto game = LoadGame("gomuko");
  auto state = game->NewInitialState();
  state->ApplyAction(20);
  state->ApplyAction(8);
  state->ApplyAction(1);
  state->ApplyAction(9);
  state->ApplyAction(2);
  state->ApplyAction(10);
  state->ApplyAction(3);
  state->ApplyAction(11);

  std::cout << state->ToString();

  SPIEL_CHECK_EQ(state->IsTerminal(), true);
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], -1);
  SPIEL_CHECK_EQ(returns[1], 1);
}

} // namespace
} // namespace gomuko
} // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::gomuko::BasicGomukoTests();
  open_spiel::gomuko::TestStateStruct();
  open_spiel::gomuko::TestObservationStruct();
  open_spiel::gomuko::TestActionStruct();
  open_spiel::gomuko::Player1Win();
  open_spiel::gomuko::Player2Win();
}
