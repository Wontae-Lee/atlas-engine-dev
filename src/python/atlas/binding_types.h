#pragma once

#include <atlas/collider/collider.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/system/system.h>
#include <atlas/unit/unit.h>

#include <utility>
#include <vector>

namespace atlas::python {

using MeshOwners = std::vector<TriangleMeshHostPtr>;

struct PyGeometry final {
    Geometry value;
    MeshOwners mesh_owners;
};

struct PyUnit final {
    Unit value;
    MeshOwners mesh_owners;
};

struct PySource final {
    SourceHostPtr value;
    MeshOwners mesh_owners;
};

struct PyCollider final {
    Collider value;
    MeshOwners mesh_owners;
};

struct PySink final {
    Sink value;
    MeshOwners mesh_owners;
};

struct PySystem final {
    PySystem(System system, MeshOwners owners)
        : mesh_owners(std::move(owners))
        , value(std::move(system)) {
    }

    MeshOwners mesh_owners;
    System value;
};

}
