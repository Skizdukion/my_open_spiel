add_executable(example_2 example_2.cc ${OPEN_SPIEL_OBJECTS})

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
      add_executable(dqn_tictactoe dqn_tictactoe.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:dqn_torch>)
    target_link_libraries (dqn_tictactoe ${TORCH_LIBRARIES})
endif()