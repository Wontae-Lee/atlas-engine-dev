from importlib import import_module as _import_module
from importlib.metadata import version as _version

from ._engine import available_engines, get_default_engine, set_default_engine


__version__ = _version("atlas-engine")

_EXPORTS = {
    "Bool3": "math",
    "Int3": "math",
    "Float3": "math",
    "Float3x3": "math",
    "Quaternion": "math",
    "GeneratorType": "generator",
    "Generator": "generator",
    "UniformGenerator": "generator",
    "JitteringGenerator": "generator",
    "MaxwellSigmaGenerator": "generator",
    "MaxwellBoltzmannGenerator": "generator",
    "Observer": "observer",
    "DiffuseSampling": "collider",
    "ColliderType": "collider",
    "Collider": "collider",
    "IsothermalCollider": "collider",
    "System": "system",
    "GeometryType": "geometry",
    "Geometry": "geometry",
    "Sphere": "geometry",
    "Plane": "geometry",
    "Box": "geometry",
    "Cylinder": "geometry",
    "Circle": "geometry",
    "Square": "geometry",
    "Triangle": "geometry",
    "PolygonalPrism": "geometry",
    "TriangleMesh": "geometry",
    "Unit": "unit",
    "Sync": "sync",
    "MaterialType": "material",
    "Material": "material",
    "MaterialDictionary": "material",
    "Molecule": "material",
    "Atom": "material",
    "Ion": "material",
    "Neutron": "material",
    "Solid": "material",
    "SinkType": "sink",
    "Sink": "sink",
    "VolumeSink": "sink",
    "SurfaceSink": "sink",
    "TracingSink": "sink",
    "DefaultRandomEngine": "random",
    "UniformRealDistribution": "random",
    "UniformRealDistributionDouble": "random",
    "Fluid": "fluid",
    "SpatialHashingSearcher": "searcher",
    "CodecType": "codec",
    "Codec": "codec",
    "KnudsenCodec": "codec",
    "DsmcKernelType": "solver",
    "DsmcKernel": "solver",
    "SolverType": "solver",
    "Solver": "solver",
    "DsmcSolver": "solver",
    "Universe": "universe",
    "SourceType": "source",
    "Source": "source",
    "VolumeSource": "source",
    "SurfaceSource": "source",
    "Ray": "spatial",
    "HitSurface": "spatial",
    "HitAABB": "spatial",
    "AABB": "spatial",
    "BVHNode": "spatial",
    "BVH": "spatial",
    "LBVH": "spatial",
    "SAHBVH": "spatial",
}

_MODULES = (
    "math",
    "generator",
    "observer",
    "collider",
    "system",
    "geometry",
    "unit",
    "sync",
    "material",
    "sink",
    "random",
    "fluid",
    "searcher",
    "codec",
    "solver",
    "universe",
    "source",
    "spatial",
    "sampling",
    "serialization",
)

__all__ = [
    "__version__",
    "available_engines",
    "get_default_engine",
    "set_default_engine",
    *_EXPORTS,
]


def __getattr__(name):
    if name in _EXPORTS:
        module = _import_module(f".{_EXPORTS[name]}", __name__)
        value = getattr(module, name)
    elif name in _MODULES:
        value = _import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_MODULES))
