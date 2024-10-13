set(BIGBRIAN_SOURCE
    "src/math/Matrix.cpp"
    "src/nn/Layer.cpp"
    "src/nn/Network.cpp"
    "src/Population.cpp"
    "src/nn/ActivationFuncs.cpp"
)

add_library(BigBrianLib
    ${BIGBRIAN_SOURCE}
)
target_include_directories(BigBrianLib PUBLIC "include/")

add_executable(BigBrianExec "src/main.cpp")
target_link_libraries(BigBrianExec BigBrianLib)
