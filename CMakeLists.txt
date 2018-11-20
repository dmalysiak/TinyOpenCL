cmake_minimum_required(VERSION 2.8)

set( SRC_FILES 
	src/TinyOpenCL.cpp 
)

add_executable(debug ${SRC_FILES})

target_link_libraries(debug
    OpenCL
)

set_property(TARGET debug PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_custom_command(TARGET debug POST_BUILD 
	COMMAND cp src/*.cl bin/
)
