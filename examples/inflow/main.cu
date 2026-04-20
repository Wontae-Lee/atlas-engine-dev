// CUDA wrapper entry point for the cylinder example.
//
// Keeping the implementation in main.cpp avoids duplicating the example logic
// between the TBB and CUDA builds while still letting CMake compile a .cu file
// when ATLAS_USE_CUDA is enabled.
#include "main.cpp"
