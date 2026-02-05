add_executable(example_2 example_2.cc ${OPEN_SPIEL_OBJECTS})

if(OPEN_SPIEL_BUILD_WITH_LIBTORCH)
    add_executable(dqn_tictactoe dqn_tictactoe.cc
        ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:dqn_torch>)
    target_link_libraries(dqn_tictactoe ${TORCH_LIBRARIES})

    add_executable(alpha_zero_example alpha_zero_example.cc
        ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:alpha_zero_torch>
        $<TARGET_OBJECTS:bots>)
    target_link_libraries(alpha_zero_example ${TORCH_LIBRARIES})
endif()
