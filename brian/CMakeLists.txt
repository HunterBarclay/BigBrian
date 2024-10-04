add_library(BigBrianLib
    "src/math/Matrix.cpp"
)
target_include_directories(BigBrianLib PUBLIC "include/")

add_executable(BigBrianExec "src/main.cpp")
target_link_libraries(BigBrianExec BigBrianLib)
