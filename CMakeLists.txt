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

find_path(STB_INCLUDE_DIRS "stb.h")
target_include_directories(${PROJECT_NAME} PRIVATE ${STB_INCLUDE_DIRS})

find_package(tinyobjloader CONFIG REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE tinyobjloader::tinyobjloader)