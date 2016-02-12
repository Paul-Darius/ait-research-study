cmake_minimum_required(VERSION 2.8)

# Name of the project

project(FaceDetector)

# Name of the output path

set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories("${PROJECT_BINARY_DIR}")

# Define sources and executable
set(EXECUTABLE_NAME "FaceDetector")

file(
        GLOB_RECURSE

        source_files

        src/*
)

add_executable(

        ${EXECUTABLE_NAME}

        ${source_files}

        )
find_package( OpenCV REQUIRED )
target_link_libraries( FaceDetector ${OpenCV_LIBS} )

# Install target
#install(TARGETS ${EXECUTABLE_NAME} DESTINATION bin)


# CPack packaging
#include(InstallRequiredSystemLibraries)
#set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/COPYING")
#set(CPACK_PACKAGE_VERSION_MAJOR "${myproject_VERSION_MAJOR}")
#set(CPACK_PACKAGE_VERSION_MINOR "${myproject_VERSION_MINOR}")
#include(CPack)

