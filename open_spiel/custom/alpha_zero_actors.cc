#include "games/gomuko/gomuko.h"

#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"
#include "utils/file.h"

#include <csignal>

ABSL_FLAG(int, actors, 7, "How many actors to run.");
ABSL_FLAG(int, num_rows, 6, "How many actors to run.");
ABSL_FLAG(int, num_cols, 6, "How many actors to run.");
ABSL_FLAG(int, win_size, 4, "How many actors to run.");
ABSL_FLAG(std::string, devices, "cpu", "How many actors to run.");

ABSL_FLAG(std::string, nn_model, "resnet", "Model type");
ABSL_FLAG(int, nn_width, 128, "Model width");
ABSL_FLAG(int, nn_depth, 10, "Model depth");

ABSL_FLAG(int, evaluation_window, 100, "Evaluation window");
ABSL_FLAG(int, batch_size, 1,
          "Inference batch size"); /* inference_batch_size */
ABSL_FLAG(int, inference_threads, 2, "Number of inference threads");

ABSL_FLAG(double, policy_alpha, 1.0, "Policy alpha");
ABSL_FLAG(double, policy_epsilon, 0.25, "Policy epsilon");

ABSL_FLAG(double, learning_rate, 0.0001, "Learning rate");
ABSL_FLAG(double, weight_decay, 0.0001, "Weight decay");
ABSL_FLAG(int, max_simulations, 300, "Max simulation");

namespace open_spiel {
namespace algorithms {
namespace torch_az {

open_spiel::StopToken stop_token;

void signal_handler(int s) {
  if (stop_token.StopRequested()) {
    exit(1);
  } else {
    stop_token.Stop();
  }
}

int run_main() {
  AlphaZeroConfig config;

  // open_spiel::GameParameters params;
  // auto game =
  // std::make_shared<open_spiel::tic_tac_toe::TicTacToeGame>(params);
  // config.game = "tic_tac_toe";

  open_spiel::GameParameters params;
  params["rows"] = GameParameter(absl::GetFlag(FLAGS_num_rows));
  params["cols"] = GameParameter(absl::GetFlag(FLAGS_num_cols));
  params["winSize"] = GameParameter(absl::GetFlag(FLAGS_win_size));

  auto game = std::make_shared<open_spiel::gomuko::GomukoGame>(params);

  config.game = "gomuko";

  config.devices = absl::GetFlag(FLAGS_devices);
  config.actors = absl::GetFlag(FLAGS_actors);

  config.path = "/home/lpk/my_open_spiel/run_output/debug_actors";
  config.graph_def = "";
  config.nn_model = absl::GetFlag(FLAGS_nn_model);
  config.nn_width = absl::GetFlag(FLAGS_nn_width);
  config.nn_depth = absl::GetFlag(FLAGS_nn_depth);
  config.explicit_learning = false;
  config.learning_rate = absl::GetFlag(FLAGS_learning_rate);
  config.weight_decay = absl::GetFlag(FLAGS_weight_decay);
  config.train_batch_size = 1 << 10;
  config.replay_buffer_size = 1 << 16;
  config.replay_buffer_reuse = 3;
  config.checkpoint_freq = 50;
  config.evaluation_window = absl::GetFlag(FLAGS_evaluation_window);
  config.uct_c = 2.0;
  config.max_simulations = absl::GetFlag(FLAGS_max_simulations);
  config.inference_batch_size = absl::GetFlag(FLAGS_batch_size);
  config.inference_threads = absl::GetFlag(FLAGS_inference_threads);
  config.inference_cache = 262144;
  config.policy_alpha = absl::GetFlag(FLAGS_policy_alpha);
  config.policy_epsilon = absl::GetFlag(FLAGS_policy_epsilon);
  config.temperature = 1;
  config.temperature_drop = 10;
  config.cutoff_probability = 0.8;
  config.cutoff_value = 0.95;
  config.evaluators = 1;
  config.eval_levels = 7;
  config.max_steps = 300;

  file::Remove(config.path);

  file::Mkdirs(config.path);
  if (!file::IsDirectory(config.path)) {
    std::cerr << config.path << " is not a directory." << std::endl;
    return false;
  }

  if (config.graph_def.empty()) {
    config.graph_def = "vpnet.pb";
    std::string model_path = absl::StrCat(config.path, "/", config.graph_def);
    if (file::Exists(model_path)) {
      std::cout << "Overwriting existing model: " << model_path << std::endl;
    } else {
      std::cout << "Creating model: " << model_path << std::endl;
    }
    SPIEL_CHECK_TRUE(CreateGraphDef(
        *game, config.learning_rate, config.weight_decay, config.path,
        config.graph_def, config.nn_model, config.nn_width, config.nn_depth));
  }

  DeviceManager device_manager;

  device_manager.AddDevice(
      VPNetModel(*game, config.path, config.graph_def, config.devices));

  open_spiel::ThreadedQueue<Trajectory> trajectory_queue(
      config.replay_buffer_size / config.replay_buffer_reuse);

  auto eval = std::make_shared<VPNetEvaluator>(
      &device_manager, config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  std::vector<Thread> actors;
  actors.reserve(config.actors);
  for (int i = 0; i < config.actors; ++i) {
    actors.emplace_back([&, i]() {
      actor(*game, config, i, &trajectory_queue, eval, &stop_token);
    });
  }

  std::signal(SIGINT, signal_handler);

  absl::Time start_time = absl::Now();
  while (!stop_token.StopRequested()) {
    absl::SleepFor(absl::Seconds(1));
  }

  double seconds = absl::ToDoubleSeconds(absl::Now() - start_time);
  int total_games = trajectory_queue.Size();
  std::cout << "Config simulatation: Devices{" << config.devices << "} Actors{"
            << config.actors << "} MapSize{" << absl::GetFlag(FLAGS_num_rows)
            << "," << absl::GetFlag(FLAGS_num_cols) << "} WinSize{"
            << absl::GetFlag(FLAGS_win_size) << "} NN_Model{" << config.nn_model
            << "} NN_Width{" << config.nn_width << "} NN_Depth{"
            << config.nn_depth << "} Inference Batch Size{"
            << config.inference_batch_size << "}\n";
  std::cout << "Total game simulated: " << total_games << ", Time: " << seconds
            << "s, Games/s: " << total_games / seconds << std::endl;

  trajectory_queue.BlockNewValues();
  trajectory_queue.Clear();

  std::cout << "Joining all the threads." << std::endl;
  for (auto &t : actors) {
    t.join();
  }
  return 0;
}
} // namespace torch_az
} // namespace algorithms
} // namespace open_spiel

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  open_spiel::algorithms::torch_az::run_main();
}