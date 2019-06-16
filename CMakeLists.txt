cmake_minimum_required(VERSION 3.5)
project(logistic_regression_sgd)

set(CMAKE_CXX_STANDARD 14)

include(${PROJECT_SOURCE_DIR}/cmake/misc.cmake)
include_directories(${PROJECT_SOURCE_DIR}/include)

configure_file(${PROJECT_SOURCE_DIR}/cmake/get_eigen.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/eigen-download/CMakeLists.txt)
setup_eigen()

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(LogisticRegression LogisticRegression.cpp LogisticRegression_learn.cpp)
add_library(LogisticRegressionHogwild LogisticRegression.cpp LogisticRegression_learn_hogwild.cpp)
add_library(LogisticRegressionHogBatch LogisticRegression.cpp LogisticRegression_learn_hogbatch.cpp)
add_library(LogisticRegressionMiniBatch LogisticRegression.cpp LogisticRegression_learn_minibatch.cpp)

target_link_libraries(LogisticRegression Eigen3::Eigen)
target_link_libraries(LogisticRegressionHogwild Eigen3::Eigen)
target_link_libraries(LogisticRegressionHogBatch Eigen3::Eigen)
target_link_libraries(LogisticRegressionMiniBatch Eigen3::Eigen)

add_executable(test1 test.cpp)
add_executable(test2 test.cpp)
add_executable(test3 test.cpp)
add_executable(test4 test.cpp)

target_link_libraries(test1 LogisticRegression)
target_link_libraries(test2 LogisticRegressionHogwild)
target_link_libraries(test3 LogisticRegressionHogBatch)
target_link_libraries(test4 LogisticRegressionMiniBatch)
