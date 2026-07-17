cmake_minimum_required(VERSION 3.10.0)
project(image-analogies VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(image-analogies 
    main.cpp 
    util/arg_parser.cpp 
    util/gaussian_pyramids.cpp
    algo/patchmatch.cpp
    algo/feature_vector.cpp
    algo/image_analogy.cpp)

list(APPEND CMAKE_PREFIX_PATH "C:/msys64/mingw64")
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
