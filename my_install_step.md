./install.sh

# 1. Debug Build (For Development & VS Code)

mkdir -p build
cd build

# Note: Point to ../open_spiel because that's where CMakeLists.txt is

BUILD_TYPE=Debug OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON OPEN_SPIEL_BUILD_WITH_LIBNOP=ON BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DBUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../open_spiel
make -j$(nproc) open_spiel

# 2. Release Build (For Performance/Benchmarking)

cd ..
mkdir -p build_release
cd build_release
BUILD_TYPE=Release OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON OPEN_SPIEL_BUILD_WITH_LIBNOP=ON BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DBUILD_TYPE=Release ../open_spiel
make -j$(nproc) open_spiel

# 3. Link build compile_commands to apply clangd language server

ln -sf build_debug/compile_commands.json compile_commands.json

SOME HELPER DEBUG:
? params
p \*params
p params->params
p params->toString()

    p auto $s = game->NewInitialState().release()
    p $s
    p $s->ToString()

    frame variable config

    x/s *(char**)&config.game # Go to this memory address and print bytes as a null-terminated C string.

Helpful command
./alpha_zero_example --game=tic_tac_toe --path=/home/lpk/my_open_spiel/run_output/alpha_zero_tic_tac_toe --devices=cuda:0 --max_steps=10

./alpha_zero_interactive /home/lpk/my_open_spiel/run_output/alpha_zero_tic_tac_toe/config.json /home/lpk/my_open_spiel/run_output/alpha_zero_tic_tac_toe/checkpoint-100

./alpha_zero_vs_random /home/lpk/my_open_spiel/run_output/alpha_zero_tic_tac_toe/config.json /home/lpk/my_open_spiel/run_output/alpha_zero_tic_tac_toe/checkpoint-100

./alpha_zero_self_play /home/lpk/my_open_spiel/run_output/alpha_zero_tic_tac_toe/config.json /home/lpk/my_open_spiel/run_output/alpha_zero_tic_tac_toe/checkpoint-100 /home/lpk/my_open_spiel/run_output/alpha_zero_tic_tac_toe/checkpoint-200 --num_games=10000

./custom/alpha_zero_example \
 --game=gomuko \
 --path=/home/lpk/my_open_spiel/run_output/alpha_zero_gomuko_basic \
 --devices=cuda:0 \
 --max_steps=500 
 --actors=6