cmake_minimum_required (VERSION 2.8.8)
project (rldm CXX)

find_package (Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if (CMAKE_COMPILER_IS_GNUCXX)
   add_definitions ("-Werror -Wall -Wextra -march=native -mno-avx")
endif()

if (CMAKE_CXX_COMPILER_ID MATECHES "Clang")
   add_definitions ("-Werror -Wall")
endif()
