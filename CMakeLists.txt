cmake_minimum_required(VERSION 3.10.0)
project(image-analogies VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(image-analogies 
    main.cpp 
    util/arg_parser.cpp 
    util/gaussian_pyramids.cpp
    algo/image_analogy.cpp)

find_package(OpenCV REQUIRED)
target_include_directories(image-analogies PRIVATE
    ${OpenCV_INCLUDE_DIRS} 
    util
    algo
)

target_link_libraries(image-analogies ${OpenCV_LIBS})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
