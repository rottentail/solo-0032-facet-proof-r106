from .algebra import point, face_normal, unit, sub, dot
from .camera import frustum_planes, project, view_point
from .clipping import clip_triangle
from .errors import require
from .shading import illuminate
from .values import integer, digest
from .pigments import srgb_to_linear

# facet-raster-1 is the legacy opaque-only definition. The first transparent
# material in a closure bumps the definition to facet-raster-2, whose frozen
# semantics are order-independent bounded over-compositing. The revision string
# travels through the request hash and tile fingerprint, so a definition change
# can never silently reuse tiles or idempotent results from the other path.
RENDERER_REVISION = "facet-raster-1"
TRANSPARENT_REVISION = "facet-raster-2"

# Per-sample named-layer budget bounds. Every retained layer has explicit,
# separately stored identity; deeper fragments are alpha-merged into one
# unnamed tail, so memory stays linear in this bound instead of fragment count.
LAYER_BUDGET_DEFAULT = 16
LAYER_BUDGET_MIN = 1
LAYER_BUDGET_MAX = 64


def validate_request(data):
    width = integer(data.get("width", 256), "width", 16, 512)
    height = integer(data.get("height", 256), "height", 16, 512)
    samples = integer(data.get("samples", 1), "samples", 1, 2)
    layer_budget = integer(data.get("layer_budget", LAYER_BUDGET_DEFAULT),
                           "layer_budget", LAYER_BUDGET_MIN, LAYER_BUDGET_MAX)
    return {"width": width, "height": height, "samples": samples,
            "layer_budget": layer_budget, "renderer": RENDERER_REVISION}


def build_plan(seal, request):
    width, height = request["width"] * request["samples"], request["height"] * request["samples"]
    lens = seal["lens"]
    planes = frustum_planes(lens, width, height)
    triangles = []
    objects = {}
    total_faces = sum(len(item["mesh"]["faces"]) for item in seal["closure"])
    require(total_faces <= 40000, "capacity", "Exposure exceeds 40000 source faces", 422)
    # A placement is transparent only through its own frozen pigment. Object
    # ids are still assigned by sorted placement key, so reordering the draft
    # placements never changes ownership; depth alone decides visibility.
    sorted_items = sorted(seal["closure"], key=lambda item: item["placement"]["key"])
    transparent = any(item["pigment"].get("opacity", 1) < 1 for item in sorted_items)
    for object_id, item in enumerate(sorted_items, 1):
        placement, mesh, pigment = item["placement"], item["mesh"], item["pigment"]
        opacity = pigment.get("opacity", 1)
        is_transparent = opacity < 1
        objects[str(object_id)] = {"key": placement["key"], "mesh_id": placement["mesh_id"],
                                   "pigment_id": placement["pigment_id"]}
        vertices = [point(placement["matrix"], v) for v in mesh["vertices"]]
        for face_number, face in enumerate(mesh["faces"]):
            world = [vertices[i] for i in face]
            normal = face_normal(world)
            center = [sum(v[a] for v in world) / 3 for a in range(3)]
            direction = sub(lens["eye"], center)
            if dot(direction, direction) < 1e-12:
                continue
            view_direction = unit(direction) if lens["projection"] == "perspective" else [-v for v in lens["forward"]]
            # Transparent shells must show their interior through both entry and
            # exit walls, so back-face culling is disabled for their materials.
            if not is_transparent and not pigment["two_sided"] and dot(normal, view_direction) <= 0:
                continue
            color = illuminate(normal, pigment, seal["illumination"], view_direction)
            camera_vertices = [view_point(lens, v) for v in world]
            for clipped in clip_triangle(camera_vertices, planes):
                triangles.append({"vertices": [project(lens, v, width, height) for v in clipped],
                                  "color": color, "object_id": object_id, "face": face_number,
                                  "opacity": opacity, "transparent": is_transparent})
    renderer = TRANSPARENT_REVISION if transparent else RENDERER_REVISION
    # The effective renderer revision is derived from the frozen closure, not
    # trusted from the caller. It is part of the request hash and tile
    # fingerprint, so transparent and opaque renders of the same nominal
    # request never share checkpoints or idempotent results.
    request["renderer"] = renderer
    plan = {"width": width, "height": height, "triangles": triangles, "objects": objects,
            "projection": lens["projection"],
            "background": [srgb_to_linear(c) for c in seal["illumination"]["background"]],
            "lens_hash": digest(lens), "source_faces": total_faces,
            "transparent": transparent, "renderer": renderer}
    if transparent:
        # The frozen semantics block: compositing operator, identity split and
        # the named-layer bound are all part of the render definition, hence the
        # tile fingerprint and idempotent request hash capture them.
        plan["layer_budget"] = request["layer_budget"]
        plan["transparency"] = {
            "model": "front_to_back_over",
            "depth_order": ["camera_forward_distance", "object_id", "face"],
            "surface_identity": "nearest_named_fragment",
            "contributor_identity": "largest_named_color_contribution",
            "overflow": "order_independent_weighted_tail",
            "overflow_detail": ("exact aggregate coverage 1-product(1-alpha) "
                                "with alpha-weighted average color; per-fragment "
                                "depth ordering and object identity are not retained"),
            "layer_budget": request["layer_budget"]}
    return plan
