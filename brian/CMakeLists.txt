set(BIGBRIAN_SOURCE
    "src/math/matrix.cpp"
    "src/nn/layer.cpp"
    "src/nn/network.cpp"
    "src/population.cpp"
    "src/nn/activation_funcs.cpp"
    "src/util/data_parser.cpp"
)

add_library(BigBrianLib
    ${BIGBRIAN_SOURCE}
)
target_include_directories(BigBrianLib PUBLIC "include/")

add_executable(BigBrianExec "src/main.cpp")
target_link_libraries(BigBrianExec BigBrianLib)
