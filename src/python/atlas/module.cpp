#include "codec/codec.h"
#include "codec/codec_type.h"
#include "codec/knudsen_codec.h"
#include "collider/collider.h"
#include "collider/collider_type.h"
#include "collider/diffuse_sampling.h"
#include "collider/isothermal_collider.h"
#include "fluid/fluid.h"
#include "generator/generator.h"
#include "generator/generator_type.h"
#include "generator/jittering_generator.h"
#include "generator/maxwell_boltzmann_generator.h"
#include "generator/maxwell_sigma_generator.h"
#include "generator/uniform_generator.h"
#include "geometry/box.h"
#include "geometry/circle.h"
#include "geometry/cylinder.h"
#include "geometry/geometry.h"
#include "geometry/geometry_type.h"
#include "geometry/plane.h"
#include "geometry/polygonal_prism.h"
#include "geometry/sphere.h"
#include "geometry/square.h"
#include "geometry/triangle.h"
#include "geometry/triangle_mesh.h"
#include "material/atom.h"
#include "material/ion.h"
#include "material/material.h"
#include "material/material_dictionary.h"
#include "material/material_type.h"
#include "material/molecule.h"
#include "material/neutron.h"
#include "material/solid.h"
#include "math/constants.h"
#include "math/math.h"
#include "math/matrix/float3x3.h"
#include "math/quaternion.h"
#include "math/vector/bool3.h"
#include "math/vector/float3.h"
#include "math/vector/int3.h"
#include "observer/observer.h"
#include "random/default_random_engine.h"
#include "random/seed.h"
#include "random/uniform_real_distribution.h"
#include "sampling/sampling.h"
#include "searcher/spatial_hashing_searcher.h"
#include "serialization/protobuf_snapshot.h"
#include "sink/sink.h"
#include "sink/sink_type.h"
#include "sink/surface_sink.h"
#include "sink/tracing_sink.h"
#include "sink/volume_sink.h"
#include "solver/dsmc/dsmc_solver.h"
#include "solver/dsmc/kernel/dsmc_kernel.h"
#include "solver/dsmc/kernel/dsmc_kernel_type.h"
#include "solver/solver.h"
#include "solver/solver_type.h"
#include "source/source.h"
#include "source/source_type.h"
#include "source/surface_source.h"
#include "source/volume_source.h"
#include "spatial/axis_aligned_bounding_box.h"
#include "spatial/bounding_volume_hierarchy/bvh.h"
#include "spatial/bounding_volume_hierarchy/lbvh.h"
#include "spatial/bounding_volume_hierarchy/node.h"
#include "spatial/bounding_volume_hierarchy/sah_bvh.h"
#include "spatial/ray.h"
#include "sync/sync.h"
#include "system/system.h"
#include "unit/unit.h"
#include "universe/universe.h"

#include <nanobind/nanobind.h>

NB_MODULE(_core, m) {
    m.attr("__version__") = ATLAS_VERSION_STRING;
    auto math             = m.def_submodule("math");
    auto random           = m.def_submodule("random");
    auto spatial          = m.def_submodule("spatial");
    auto sync             = m.def_submodule("sync");
    auto geometry         = m.def_submodule("geometry");
    auto unit             = m.def_submodule("unit");
    auto material         = m.def_submodule("material");
    auto fluid            = m.def_submodule("fluid");
    auto universe         = m.def_submodule("universe");
    auto source           = m.def_submodule("source");
    auto generator        = m.def_submodule("generator");
    auto solver           = m.def_submodule("solver");
    auto codec            = m.def_submodule("codec");
    auto collider         = m.def_submodule("collider");
    auto sink             = m.def_submodule("sink");
    auto observer         = m.def_submodule("observer");
    auto system           = m.def_submodule("system");
    auto searcher         = m.def_submodule("searcher");
    auto sampling         = m.def_submodule("sampling");
    auto serialization    = m.def_submodule("serialization");

    atlas::python::math::register_bool3(math);
    atlas::python::math::register_int3(math);
    atlas::python::math::register_float3(math);
    atlas::python::math::register_float3x3(math);
    atlas::python::math::register_quaternion(math);
    atlas::python::math::register_constants(math);
    atlas::python::math::register_math(math);
    atlas::python::register_default_random_engine(random);
    atlas::python::register_uniform_real_distribution(random);
    atlas::python::register_seed(random);
    atlas::python::register_ray(spatial);
    atlas::python::register_sync(sync);
    atlas::python::register_axis_aligned_bounding_box(spatial);
    atlas::python::register_geometry_type(geometry);
    atlas::python::register_geometry(geometry);
    atlas::python::register_sphere(geometry);
    atlas::python::register_plane(geometry);
    atlas::python::register_box(geometry);
    atlas::python::register_cylinder(geometry);
    atlas::python::register_circle(geometry);
    atlas::python::register_square(geometry);
    atlas::python::register_triangle(geometry);
    atlas::python::register_polygonal_prism(geometry);
    atlas::python::register_triangle_mesh(geometry);
    atlas::python::register_node(spatial);
    atlas::python::register_bvh(spatial);
    atlas::python::register_lbvh(spatial);
    atlas::python::register_sah_bvh(spatial);
    atlas::python::register_unit(unit);
    atlas::python::register_material_type(material);
    atlas::python::register_material(material);
    atlas::python::register_molecule(material);
    atlas::python::register_atom(material);
    atlas::python::register_ion(material);
    atlas::python::register_neutron(material);
    atlas::python::register_solid(material);
    atlas::python::register_material_dictionary(material);
    atlas::python::register_fluid(fluid);
    atlas::python::register_universe(universe);
    atlas::python::register_source_type(source);
    atlas::python::register_source(source);
    atlas::python::register_volume_source(source);
    atlas::python::register_surface_source(source);
    atlas::python::register_generator_type(generator);
    atlas::python::register_generator(generator);
    atlas::python::register_maxwell_boltzmann_generator(generator);
    atlas::python::register_uniform_generator(generator);
    atlas::python::register_jittering_generator(generator);
    atlas::python::register_maxwell_sigma_generator(generator);
    atlas::python::register_dsmc_kernel_type(solver);
    atlas::python::register_dsmc_kernel(solver);
    atlas::python::register_solver_type(solver);
    atlas::python::register_solver(solver);
    atlas::python::register_dsmc_solver(solver);
    atlas::python::register_codec_type(codec);
    atlas::python::register_codec(codec);
    atlas::python::register_knudsen_codec(codec);
    atlas::python::register_diffuse_sampling(collider);
    atlas::python::register_collider_type(collider);
    atlas::python::register_collider(collider);
    atlas::python::register_isothermal_collider(collider);
    atlas::python::register_sink_type(sink);
    atlas::python::register_sink(sink);
    atlas::python::register_volume_sink(sink);
    atlas::python::register_surface_sink(sink);
    atlas::python::register_tracing_sink(sink);
    atlas::python::register_observer(observer);
    atlas::python::register_system(system);
    atlas::python::register_spatial_hashing_searcher(searcher);
    atlas::python::register_sampling(sampling);
    atlas::python::register_protobuf_snapshot(serialization);
}
