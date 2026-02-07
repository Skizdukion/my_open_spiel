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

#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace gomuko {
namespace {

void RunBenchMarkTicTacToe(int num_games) {
  std::cout << "Starting benchmark: TicTacToe";

  std::shared_ptr<const Game> game = LoadGame("tic_tac_toe");

  std::mt19937 rng(42);

  auto start_time = std::chrono::high_resolution_clock::now();

  long long total_moves = 0;

  for (int i = 0; i < num_games; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      std::vector<Action> legal_actions = state->LegalActions();
      if (legal_actions.empty())
        break;

      // Simple uniform random selection
      std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
      Action action = legal_actions[dist(rng)];
      state->ApplyAction(action);
      total_moves++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> distinct = end_time - start_time;

  std::cout << "Finished " << num_games << " games in " << distinct.count()
            << " seconds." << std::endl;
  std::cout << "Average time per game: "
            << (distinct.count() * 1000.0) / num_games << " ms" << std::endl;
  std::cout << "Average moves per game: " << (double)total_moves / num_games
            << std::endl;
  std::cout << "Moves per second: " << total_moves / distinct.count()
            << std::endl;
  std::cout << "--------------------------------------------------"
            << std::endl;
}

void RunBenchmark(const std::string &label, int rows, int cols, int win_size,
                  int num_games) {
  std::cout << "Starting benchmark: " << label << " (" << rows << "x" << cols
            << ", win=" << win_size << ")" << std::endl;

  GameParameters params;
  params["rows"] = GameParameter(rows);
  params["cols"] = GameParameter(cols);
  params["winSize"] = GameParameter(win_size);

  std::shared_ptr<const Game> game = LoadGame("gomuko", params);

  std::mt19937 rng(42); // Fixed seed for reproducibility

  auto start_time = std::chrono::high_resolution_clock::now();

  long long total_moves = 0;

  for (int i = 0; i < num_games; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      std::vector<Action> legal_actions = state->LegalActions();
      if (legal_actions.empty())
        break;

      // Simple uniform random selection
      std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
      Action action = legal_actions[dist(rng)];
      state->ApplyAction(action);
      total_moves++;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> distinct = end_time - start_time;

  std::cout << "Finished " << num_games << " games in " << distinct.count()
            << " seconds." << std::endl;
  std::cout << "Average time per game: "
            << (distinct.count() * 1000.0) / num_games << " ms" << std::endl;
  std::cout << "Average moves per game: " << (double)total_moves / num_games
            << std::endl;
  std::cout << "Moves per second: " << total_moves / distinct.count()
            << std::endl;
  std::cout << "--------------------------------------------------"
            << std::endl;
}

} // namespace
} // namespace gomuko
} // namespace open_spiel

int main(int argc, char **argv) {
  int num_games = 10000000;

  open_spiel::gomuko::RunBenchMarkTicTacToe(num_games);
  open_spiel::gomuko::RunBenchmark("Small 3x3", 3, 3, 3, num_games);
  open_spiel::gomuko::RunBenchmark("Medium 5x5", 6, 6, 4, num_games);
  // open_spiel::gomuko::RunBenchmark("Large 16x16", 16, 16, 5, num_games);
}
