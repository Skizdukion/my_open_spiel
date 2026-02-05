// Copyright 2021 DeepMind Technologies Limited
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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/init.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {
namespace {

void PlayGame(const Game &game, std::shared_ptr<Bot> bot, int human_player) {
  std::unique_ptr<State> state = game.NewInitialState();
  while (!state->IsTerminal()) {
    Player current_player = state->CurrentPlayer();
    Action action;

    if (current_player == human_player) {
      std::vector<Action> legal_actions = state->LegalActions();
      std::cout << "Your turn (Player " << (human_player == 0 ? "X" : "O")
                << "). "
                << "Enter coordinates (e.g., '0 0'): ";

      std::cout << "\nCurrent Game State:\n" << state->ToString();

      std::cout << "\nAvaliable action:\n";
      for (auto a : legal_actions) {
        std::cout << state->ActionToString(a) << " ";
      }

      std::string action_str;
      while (true) {
        if (!std::getline(std::cin, action_str))
          break;
        if (action_str.empty())
          continue;

        std::optional<Action> action_opt =
            state->StringToActionSafe(action_str);

        if (!action_opt.has_value()) {
          std::cout << "Invalid action, Try again. Legal actions are: ";
          for (auto a : legal_actions) {
            std::cout << state->ActionToString(a) << " ";
          }
          std::cout << std::endl;
        } else {
          action = action_opt.value();
          break;
        }
      }
    } else {
      std::cout << "Bot is thinking..." << std::endl;
      action = bot->Step(*state);
      std::cout << "Bot played: "
                << state->ActionToString(current_player, action) << std::endl;
    }

    state->ApplyAction(action);
  }

  std::cout << "\nGame Over!\n" << state->ToString() << std::endl;
  std::vector<double> returns = state->Returns();
  if (returns[human_player] > 0)
    std::cout << "You win!" << std::endl;
  else if (returns[human_player] < 0)
    std::cout << "You lose!" << std::endl;
  else
    std::cout << "Draw!" << std::endl;
}

} // namespace
} // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::Init("", &argc, &argv, true);
  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);

  if (positional_args.size() != 3) {
    std::cerr << "Usage: " << positional_args[0]
              << " <config_json_path> <checkpoint_pt_path>" << std::endl;
    return 1;
  }

  std::string config_path = positional_args[1];
  std::string checkpoint_path = positional_args[2];

  // Load config
  open_spiel::file::File config_file(config_path, "r");
  open_spiel::json::Object config_json =
      open_spiel::json::FromString(config_file.ReadContents())
          .value()
          .GetObject();

  open_spiel::algorithms::torch_az::AlphaZeroConfig config;
  config.FromJson(config_json);

  // Load Game
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(config.game);

  // Load Model
  open_spiel::algorithms::torch_az::DeviceManager device_manager;
  device_manager.AddDevice(open_spiel::algorithms::torch_az::VPNetModel(
      *game, config.path, config.graph_def, config.devices));

  // Create Evaluator
  auto eval =
      std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(
          &device_manager, config.inference_batch_size,
          config.inference_threads, config.inference_cache, 1);

  // Load Checkpoint
  device_manager.Get(0)->LoadCheckpoint(checkpoint_path);

  // Setup Bot
  auto bot = std::make_unique<open_spiel::algorithms::MCTSBot>(
      *game, eval, config.uct_c, config.max_simulations,
      /*max_memory_mb=*/1000,
      /*solve=*/false,
      /*seed=*/0,
      /*verbose=*/false, open_spiel::algorithms::ChildSelectionPolicy::PUCT,
      /*dirichlet_alpha=*/0,
      /*dirichlet_epsilon=*/0,
      /*dont_return_chance_node=*/true);

  // Game Setup
  std::cout << "Game: " << config.game << std::endl;
  std::cout << "Loaded checkpoint from: " << checkpoint_path << std::endl;

  int human_player = -1;
  while (human_player != 0 && human_player != 1) {
    std::cout << "Choose your side (0 for X, 1 for O): ";
    std::cin >> human_player;
    // Clear newline from buffer
    std::string dummy;
    std::getline(std::cin, dummy);
  }

  open_spiel::PlayGame(*game, std::move(bot), human_player);

  return 0;
}
