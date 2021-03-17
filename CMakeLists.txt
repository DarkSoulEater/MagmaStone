cmake_minimum_required(VERSION 3.17)
project(MagmaStone)

set(CMAKE_CXX_STANDARD 17)

add_executable(MagmaStone main.cpp)

find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} glfw)

find_package(glm CONFIG REQUIRED)