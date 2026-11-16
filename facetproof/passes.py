import math
import struct
from .pigments import linear_to_srgb
from .png import encode_rgb

PLATE_FORMAT_V1 = "facet-plate-1"
PLATE_FORMAT_V2 = "facet-plate-2"
# Distinct preview tints keep the two identity semantics visually separable:
# object.png uses the legacy factors, contributor.png a different palette.
OBJECT_FACTORS = (73, 151, 199)
CONTRIBUTOR_FACTORS = (199, 121, 73)


def _nearest_sample(plan, rendered, offsets):
    return min(offsets, key=lambda p: (rendered["depth"][p], p))


def serialize_opaque(plan, rendered, request):
    """Legacy facet-plate-1 serialization; kept byte-for-byte unchanged."""
    width, height, samples = request["width"], request["height"], request["samples"]
    rgb = bytearray(width * height * 3)
    depth = bytearray(width * height * 4)
    ids = bytearray(width * height * 4)
    depth_rgb = bytearray(width * height * 3)
    object_rgb = bytearray(width * height * 3)
    coverage = 0
    for y in range(height):
        for x in range(width):
            offsets = [(y*samples+sy)*plan["width"] + x*samples+sx
                       for sy in range(samples) for sx in range(samples)]
            nearest = _nearest_sample(plan, rendered, offsets)
            obj_id = rendered["ids"][nearest]
            distance = rendered["depth"][nearest]
            output = y * width + x
            struct.pack_into("<f", depth, output*4, distance)
            struct.pack_into("<I", ids, output*4, obj_id)
            for channel in range(3):
                value = sum(rendered["colors"][p*3+channel] for p in offsets) / len(offsets)
                rgb[output*3+channel] = round(linear_to_srgb(value) * 255)
                object_rgb[output*3+channel] = ((obj_id * OBJECT_FACTORS[channel]) % 255) if obj_id else 0
                shade = round(255 / (1 + distance)) if math.isfinite(distance) else 0
                depth_rgb[output*3+channel] = shade
            coverage += bool(obj_id)
    passes = {"color.rgb": bytes(rgb), "depth.f32": bytes(depth), "object.u32": bytes(ids),
              "color.png": encode_rgb(width, height, bytes(rgb)),
              "depth.png": encode_rgb(width, height, bytes(depth_rgb)),
              "object.png": encode_rgb(width, height, bytes(object_rgb))}
    statistics = {"covered_pixels": coverage, "triangles": rendered["triangles"],
                  "fragments": rendered["fragments"], "background_depth": "positive_infinity",
                  "depth_units": "camera_forward_distance", "byte_order": "little"}
    return passes, statistics


def _preview_id(index, factors):
    return bytes(((index * factor) % 255) if index else 0 for factor in factors)


def serialize_transparent(plan, rendered, request):
    """facet-plate-2: bounded transparency with two identities and loss flags."""
    width, height, samples = request["width"], request["height"], request["samples"]
    high_width = plan["width"]
    rgb = bytearray(width * height * 3)
    depth = bytearray(width * height * 4)
    surface = bytearray(width * height * 4)
    contributor = bytearray(width * height * 4)
    depth_rgb = bytearray(width * height * 3)
    surface_rgb = bytearray(width * height * 3)
    contributor_rgb = bytearray(width * height * 3)
    layers_out = bytearray(width * height * 2)
    truncation_out = bytearray(width * height)
    coverage = 0
    truncated_pixels = 0
    truncated_samples = 0
    overflow_total = 0
    max_layers = 0
    for y in range(height):
        for x in range(width):
            offsets = [(y*samples+sy)*high_width + x*samples+sx
                       for sy in range(samples) for sx in range(samples)]
            output = y * width + x
            # Surface identity and depth follow the nearest high sample, the
            # same nearest-sample rule the opaque pass uses.
            nearest = _nearest_sample(plan, rendered, offsets)
            surface_id = rendered["ids"][nearest]
            distance = rendered["depth"][nearest]
            # Contributor identity is the object with the largest summed color
            # contribution across the samples, never reused from the surface.
            totals = {}
            for p in offsets:
                cid = rendered["contributors"][p]
                if cid:
                    totals[cid] = totals.get(cid, 0.) + rendered["contribution"][p]
            contributor_id = max(totals, key=lambda cid: (totals[cid], -cid)) if totals else 0
            observed = max(rendered["observed_layers"][p] for p in offsets)
            is_truncated = any(rendered["truncated"][p] for p in offsets)
            struct.pack_into("<f", depth, output*4, distance)
            struct.pack_into("<I", surface, output*4, surface_id)
            struct.pack_into("<I", contributor, output*4, contributor_id)
            struct.pack_into("<H", layers_out, output*2, observed)
            truncation_out[output] = 1 if is_truncated else 0
            for channel in range(3):
                value = sum(rendered["colors"][p*3+channel] for p in offsets) / len(offsets)
                rgb[output*3+channel] = round(linear_to_srgb(value) * 255)
                depth_rgb[output*3+channel] = (round(255 / (1 + distance))
                                               if math.isfinite(distance) else 0)
            surface_rgb[output*3:output*3+3] = _preview_id(surface_id, OBJECT_FACTORS)
            contributor_rgb[output*3:output*3+3] = _preview_id(contributor_id, CONTRIBUTOR_FACTORS)
            coverage += bool(surface_id)
            truncated_pixels += int(is_truncated)
            truncated_samples += sum(1 for p in offsets if rendered["truncated"][p])
            overflow_total += sum(rendered["overflow_fragments"][p] for p in offsets)
            max_layers = max(max_layers, observed)
    passes = {"color.rgb": bytes(rgb), "depth.f32": bytes(depth),
              "object.u32": bytes(surface), "contributor.u32": bytes(contributor),
              "layers.u16": bytes(layers_out), "truncated.u8": bytes(truncation_out),
              "color.png": encode_rgb(width, height, bytes(rgb)),
              "depth.png": encode_rgb(width, height, bytes(depth_rgb)),
              "object.png": encode_rgb(width, height, bytes(surface_rgb)),
              "contributor.png": encode_rgb(width, height, bytes(contributor_rgb))}
    statistics = {"covered_pixels": coverage, "triangles": rendered["triangles"],
                  "fragments": rendered["fragments"], "background_depth": "positive_infinity",
                  "depth_units": "camera_forward_distance", "byte_order": "little",
                  "transparency": plan["transparency"],
                  "layer_budget": plan["layer_budget"],
                  "observed_layer_max": max_layers,
                  "truncated_samples": truncated_samples,
                  "truncated_pixels": truncated_pixels,
                  "overflow_fragments": overflow_total}
    return passes, statistics


def serialize_passes(plan, rendered, request):
    if plan.get("transparent"):
        return serialize_transparent(plan, rendered, request)
    return serialize_opaque(plan, rendered, request)
