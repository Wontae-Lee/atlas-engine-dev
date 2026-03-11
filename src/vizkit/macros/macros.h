#pragma once
#ifdef ATLAS_ENABLE_VIZKIT
#ifdef ATLAS_TASKING_CUDA
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#else
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <atlas/atlas.h>
#endif
#endif
