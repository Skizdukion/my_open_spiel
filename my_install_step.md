./install.sh
# 1. Debug Build (For Development & VS Code)
mkdir -p build_debug
cd build_debug
export BUILD_TYPE=Debug
export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON 
export OPEN_SPIEL_BUILD_WITH_LIBNOP=ON 
# Note: Point to ../open_spiel because that's where CMakeLists.txt is
BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DBUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../open_spiel
make -j$(nproc) open_spiel

# 2. Release Build (For Performance/Benchmarking)
cd ..
mkdir -p build_release
cd build_release
export BUILD_TYPE=Release
export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON 
export OPEN_SPIEL_BUILD_WITH_LIBNOP=ON 
BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DBUILD_TYPE=Release ../open_spiel
make -j$(nproc) open_spiel


# 3. Link build compile_commands to apply clangd language server
ln -sf build_debug/compile_commands.json compile_commands.json

SOME HELPER DEBUG:
    ? params
    p *params
    p params->params
    p params->toString()

    p auto $s = game->NewInitialState().release()
    p $s
    p $s->ToString()

    frame variable config

    x/s *(char**)&config.game # Go to this memory address and print bytes as a null-terminated C string.