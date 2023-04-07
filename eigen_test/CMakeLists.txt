cmake_minimum_required(VERSION 2.8.3)
project(first_program)

find_package(Eigen3 REQUIRED)

include_directories(${Eigen3_INCLUDE_DIR})

add_executable(${PROJECT_NAME} first_program.cpp)
target_link_libraries(${PROJECT_NAME} ${Eigen3_LIBRARIES})

add_executable(eigen_dynamic eigen_dynamic.cpp)
target_link_libraries(eigen_dynamic ${Eigen3_LIBRARIES})