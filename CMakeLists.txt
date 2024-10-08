cmake_minimum_required(VERSION 3.20)

set (CMAKE_CXX_STANDARD 14)

project(BigBrian VERSION 1.0.0)

include(CTest)

add_subdirectory("./brian")

### UNIT TESTING
# So this was a process, but essentially for ease of compatibility with IDE
# debuggers, I'm making each unit test it's own executable, all stored in a
# similar fashion that they are when they are compiled execept inside the
# output directory.

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
    "test/src/network/feedforward.cpp"
    "test/src/network/score.cpp"
)

set(UtilSource "test/src/util.cpp")

foreach(test ${TestsToRun})
    get_filename_component(TestName ${test} NAME_WE)
    # string(REGEX REPLACE "/[^/]*$" "" Dir ${test})
    string(REGEX REPLACE "^test/src/" "" RemoveTestDir ${test})
    if (RemoveTestDir MATCHES "/")
        string(REGEX REPLACE "/[^/]*$" "" TestDirPath ${RemoveTestDir})
        string(REGEX REPLACE "/" "_" TestDirName ${TestDirPath})
        set(TestTarget ${TestDirName}_${TestName})
    else()
        set(TestDirPath "")
        set(TestTarget ${TestName})
    endif()
    
    message("Adding Test: ${TestTarget}")

    add_executable(${TestTarget} ${test} ${UtilSource})
    set_target_properties(${TestTarget} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}tests/${TestDirPath}/")
    target_link_libraries(${TestTarget} BigBrianLib)
    target_include_directories(${TestTarget} PRIVATE "test/include/")

    add_test(NAME ${TestTarget} COMMAND ${TestTarget})
endforeach()
