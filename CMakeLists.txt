cmake_minimum_required(VERSION 3.20)

set (CMAKE_CXX_STANDARD 14)

project(BigBrian VERSION 1.0.0)

include(CTest)

add_subdirectory("./brian")

set(TestsToRun
    "test/src/sample.cpp"

    "test/src/math/matrix/init.cpp"
    "test/src/math/matrix/mult.cpp"
    "test/src/math/matrix/mult_reuse.cpp"
    "test/src/math/matrix/add.cpp"
    "test/src/math/matrix/add_reuse.cpp"
    "test/src/math/matrix/str.cpp"
    "test/src/math/matrix/mutate.cpp"

    "test/src/network/init.cpp"
)
create_test_sourcelist(Tests "test/src/common.cpp" ${TestsToRun})

add_executable(CommonTest "test/src/util.cpp" ${Tests})
set(TEST_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}tests/")
set_target_properties(${CommonTest} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR})

target_link_libraries(CommonTest BigBrianLib)
target_include_directories(CommonTest PRIVATE "test/include")

foreach(test ${TestsToRun})
    string(REGEX REPLACE "\\.[^.]*$" "" TName ${test})
    # get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND CommonTest ${TName})
endforeach()
