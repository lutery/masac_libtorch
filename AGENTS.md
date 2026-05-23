# AGENTS.md

## Build

```shell
mkdir -p build && cd build
cmake ..
make -j
```

`CMAKE_PREFIX_PATH` in `CMakeLists.txt:14` hardcodes the LibTorch path (`/Users/abc/libtorch/libtorch`). Edit this and the Eigen3 include path (`/opt/homebrew/include/eigen3` on line 17) to match your machine before building.

## Run

```shell
cd build && ./train_sac
```

Output CSV/txt go to `../data/` (relative to the build directory).

## Project Architecture

Single-flat-dir C++17 repo. No src/include split.

- `TrainSAC.cpp` — the only executable target (`train_sac` in CMake), built from `TrainSAC.cpp + ReplayBuffer.cpp`.
- `MultiSoftActorCritic.h` — the MASAC algorithm (CTDE). Each agent has its own Actor, Twin-Q critics (centralized), and learnable temperature.
- `MultiAgentEnvironment.h` — simple 2D navigation env with 3 agents, goal-reaching rewards, shared done.
- `Models.h` — Actor (16→32 hidden) and Critic (32→32 hidden) network implementations using `torch::nn::Module`.
- `ReplayBuffer.h/.cpp` — circular buffer with random sampling.
- `SoftActorCritic.h` / `TestEnvironment.h` — single-agent SAC variant; **not used by the current training loop**.
- `multiplot.py` — gif visualization from CSV logs. `plot_reward.py` — static reward plots.

## Gotchas

- **Hardcoded paths**: `CMakeLists.txt` must be edited for LibTorch and Eigen3 locations on your machine.
- **`TestPPO.cpp`**: listed in `CMakeLists.txt:23` `SOURCES` var but does not exist. Not needed for the executale; safe to ignore or remove from the list.
- **Build type is Debug** (`-g -O2`) with `CMAKE_EXPORT_COMPILE_COMMANDS ON` for IDE support.
- **No tests, no CI, no formatter/linter** configured in this repo.
