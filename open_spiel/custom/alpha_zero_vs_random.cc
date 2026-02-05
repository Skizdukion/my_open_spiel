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

#include <ctime>
#include <iostream>
#include <memory>
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
#include "open_spiel/spiel_bots.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/init.h"
#include "open_spiel/utils/json.h"

ABSL_FLAG(int, num_games, 1000, "Number of games to play.");
ABSL_FLAG(int, az_player, -1,
          "Player ID for AlphaZero (0 or 1). Set to -1 to alternate.");

namespace open_spiel {
namespace {

int PlayOneGame(const Game &game, Bot *bot0, Bot *bot1) {
  std::unique_ptr<State> state = game.NewInitialState();
  while (!state->IsTerminal()) {
    Player current_player = state->CurrentPlayer();
    Action action;
    if (current_player == 0) {
      action = bot0->Step(*state);
    } else {
      action = bot1->Step(*state);
    }
    state->ApplyAction(action);
  }

  std::vector<double> returns = state->Returns();
  if (returns[0] > 0)
    return 0; // Player 0 wins
  if (returns[1] > 0)
    return 1; // Player 1 wins
  return -1;  // Draw
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
  int num_games = absl::GetFlag(FLAGS_num_games);
  int az_player_flag = absl::GetFlag(FLAGS_az_player);

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

  // Setup AZ Bot
  auto az_bot = std::make_unique<open_spiel::algorithms::MCTSBot>(
      *game, eval, config.uct_c, config.max_simulations,
      /*max_memory_mb=*/1000,
      /*solve=*/false,
      /*seed=*/0,
      /*verbose=*/false, open_spiel::algorithms::ChildSelectionPolicy::PUCT,
      /*dirichlet_alpha=*/0,
      /*dirichlet_epsilon=*/0,
      /*dont_return_chance_node=*/true);

  // Setup Random Bot
  // We create one for each role since MakeUniformRandomBot takes a player_id,
  // though typically it might represent "a bot that can play as player_id".
  int seed = time(nullptr);
  auto random_bot_p0 = open_spiel::MakeUniformRandomBot(0, seed);
  auto random_bot_p1 = open_spiel::MakeUniformRandomBot(1, seed);

  int az_wins = 0;
  int random_wins = 0;
  int draws = 0;

  std::cout << "Starting " << num_games << " games..." << std::endl;
  std::cout << "AlphaZero vs Random" << std::endl;

  for (int i = 0; i < num_games; ++i) {
    int current_az_player = az_player_flag;
    if (az_player_flag == -1) {
      current_az_player = i % 2;
    }

    open_spiel::Bot *bot0;
    open_spiel::Bot *bot1;

    if (current_az_player == 0) {
      bot0 = az_bot.get();
      bot1 = random_bot_p1.get();
    } else {
      bot0 = random_bot_p0.get();
      bot1 = az_bot.get();
    }

    int winner = open_spiel::PlayOneGame(*game, bot0, bot1);

    if (winner == current_az_player) {
      az_wins++;
    } else if (winner != -1) {
      random_wins++;
    } else {
      draws++;
    }

    if ((i + 1) % 10 == 0) {
      std::cout << "\rPlayed " << (i + 1) << "/" << num_games
                << " | AZ Wins: " << az_wins
                << " | Random Wins: " << random_wins << " | Draws: " << draws
                << std::flush;
    }
  }

  std::cout << std::endl;
  std::cout << "Final Results:" << std::endl;
  std::cout << "Total Games: " << num_games << std::endl;
  std::cout << "AlphaZero Wins: " << az_wins << " ("
            << (100.0 * az_wins / num_games) << "%)" << std::endl;
  std::cout << "Random Wins: " << random_wins << " ("
            << (100.0 * random_wins / num_games) << "%)" << std::endl;
  std::cout << "Draws: " << draws << " (" << (100.0 * draws / num_games) << "%)"
            << std::endl;

  return 0;
}
