cmake_minimum_required(VERSION 3.20)

set (CMAKE_CXX_STANDARD 14)

project(BigBrian VERSION 1.0.0)

include(CTest)

add_subdirectory("./brian")

# create the testing file and list of tests
set(TestsToRun
    "test/src/SampleTest.cpp"
    "test/src/MatrixTest.cpp"
)
# create_test_sourcelist(Tests "CommonTest.cpp" ${TestToRun})

# add the executable
# add_executable(CommonTest ${Tests})

# target_link_libraries(CommonTest BigBrianLib)

# Add all the ADD_TEST for each test
foreach(test ${TestsToRun})
    get_filename_component(TName ${test} NAME_WE)

    message("Adding test ${TName}")

    add_executable(${TName} ${test})
    target_link_libraries(${TName} BigBrianLib)
    set_target_properties(${TName} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}tests/")
    add_test(NAME ${TName} COMMAND ${TName})
endforeach()
