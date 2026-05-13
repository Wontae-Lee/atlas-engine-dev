foreach (ATLAS_CYLINDER_EXAMPLE IN ITEMS continum rarefied_gas)
    if (ATLAS_USE_CUDA)
        add_executable(cylinder_${ATLAS_CYLINDER_EXAMPLE}
                ${ATLAS_CYLINDER_EXAMPLE}/main.cu
        )
    else ()
        add_executable(cylinder_${ATLAS_CYLINDER_EXAMPLE}
                ${ATLAS_CYLINDER_EXAMPLE}/main.cpp
        )
    endif ()

    target_link_libraries(cylinder_${ATLAS_CYLINDER_EXAMPLE} PRIVATE atlas-core)
endforeach ()
