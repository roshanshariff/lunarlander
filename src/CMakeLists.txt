cmake_minimum_required (VERSION 2.8.8)
project (rldm CXX)

add_library (lib_simulator OBJECT simulator.cpp)
add_library (lib_tile_coder OBJECT tile_coder.cpp)
add_library (lib_policy_gradient_agent OBJECT policy_gradient_agent.cpp)
add_library (lib_utility OBJECT utility.cpp)

add_executable (main main.cpp
               $<TARGET_OBJECTS:lib_simulator>
               $<TARGET_OBJECTS:lib_tile_coder>
               $<TARGET_OBJECTS:lib_policy_gradient_agent>
               $<TARGET_OBJECTS:lib_utility>)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

find_package (Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package (Boost 1.46.0 COMPONENTS random REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries (main ${Boost_LIBRARIES})

if (CMAKE_COMPILER_IS_GNUCXX)
   add_definitions ("-Werror -Wall -Wextra -march=native -mno-avx")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   add_definitions ("-Werror -Wall")
endif()
