#include <atlas/atlas.h>
#include <fstream>

using namespace atlas;

int
main() {

    ParticleData<float> pdata(1000);
    ParticleSystem<float> psystem;

    TriangleMesh<float> triangle_mesh;
    const std::string remover_obj_path = std::string(RESOURCES_DIR) + "/box1.obj";
    const bool ok                      = triangle_mesh.load_from_obj(remover_obj_path);
    if (!ok) {
        std::cerr << "[Error] Failed to read OBJ: " << remover_obj_path
                  << "\nMake sure the file exists under RESOURCES_DIR.\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}