export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON 
export OPEN_SPIEL_BUILD_WITH_LIBNOP=ON 

./install.sh
# 1. Debug Build (For Development & VS Code)
mkdir -p build_debug
cd build_debug
export BUILD_TYPE=Debug
# Note: Point to ../open_spiel because that's where CMakeLists.txt is
BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DBUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../open_spiel
make -j$(nproc) open_spiel

# 2. Release Build (For Performance/Benchmarking)
cd ..
mkdir -p build_release
cd build_release
export BUILD_TYPE=Release
BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DBUILD_TYPE=Release ../open_spiel
make -j$(nproc) open_spiel


SOME HELPER DEBUG:
    ? params
    p *params
    p params->params
    p params->toString()