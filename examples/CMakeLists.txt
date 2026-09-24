# Native examples are enabled by ATLAS_EXAMPLES.
# Python examples import the installed package and need no CMake target.

add_subdirectory(cpp)

# The windowed example requires both Interactive execution and rendering.
if (ATLAS_INTERACTIVE AND ATLAS_INTERACTIVE_RENDERING)
    add_subdirectory(interactive)
endif ()
