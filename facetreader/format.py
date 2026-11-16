"""Versioned plate semantics, independent of the producer's Python code."""
import hashlib
import json
import math
import re
import struct
from .archive import CHANNELS_V1, CHANNELS_V2, MAX_SIDE, read_members
from .errors import InvalidBundle, require
from .png import decode_rgb

PLATE_FORMAT_V1 = "facet-plate-1"
PLATE_FORMAT_V2 = "facet-plate-2"
RAW_STRIDE = {"color.rgb": 3, "depth.f32": 4, "object.u32": 4,
              "contributor.u32": 4, "layers.u16": 2, "truncated.u8": 1}
OBJECT_FACTORS = (73, 151, 199)
CONTRIBUTOR_FACTORS = (199, 121, 73)

TRUST = {
    "verified": "Internal byte integrity and metadata/channel consistency only",
    "external_authenticity": "Not established: an attacker can replace content and all bundle-local hashes",
    "review_validity": "Unknown: the ZIP contains no proof records or live quarantine state",
    "render_correctness": "Not established: channels are decoded and cross-checked, not independently re-rendered",
}


def _object(value, field):
    require(isinstance(value, dict), "metadata_shape", field + " must be an object")
    return value


def _integer(value, low, high, field):
    require(type(value) is int and low <= value <= high, "metadata_value", field + " is out of range")
    return value


def _text(value, field):
    require(isinstance(value, str) and bool(value), "metadata_value", field + " must be text")
    return value


def _checksum(value, field):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "metadata_hash", field + " must be a lowercase SHA-256")
    return value


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise InvalidBundle("metadata_json", "Metadata is not finite canonical JSON") from exc


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _json(raw, field):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "metadata_json", "Duplicate JSON object key")
            value[key] = item
        return value
    def constant(_):
        raise InvalidBundle("metadata_json", "Non-finite JSON constant")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        canonical(value)  # Also catches overflowing exponents and bad surrogates.
    except InvalidBundle:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InvalidBundle("metadata_json", "Invalid " + field) from exc
    return _object(value, field)


def _objects(seal, objects):
    closure = seal.get("closure")
    require(isinstance(closure, list) and 1 <= len(closure) <= 80,
            "closure", "Expected a nonempty frozen placement closure")
    by_key = {}
    for entry in closure:
        entry = _object(entry, "closure entry")
        placement = _object(entry.get("placement"), "placement")
        key = _text(placement.get("key"), "placement.key")
        require(key not in by_key, "closure", "Frozen placement keys must be unique")
        require(placement.get("visible") is True, "closure", "Frozen closure must contain visible placements")
        _object(entry.get("mesh"), "frozen mesh")
        _object(entry.get("pigment"), "frozen pigment")
        for field in ("mesh_revision", "pigment_revision"):
            _integer(entry.get(field), 1, 2**63-2, field)
        by_key[key] = {"key": key,
                       "mesh_id": _text(placement.get("mesh_id"), "placement.mesh_id"),
                       "pigment_id": _text(placement.get("pigment_id"), "placement.pigment_id")}
    expected = {str(i): by_key[key] for i, key in enumerate(sorted(by_key), 1)}
    require(_object(objects, "objects") == expected, "object_mapping", "Object mapping disagrees with the frozen closure")
    return expected


def evaluate(raw):
    members = read_members(raw)
    plate = _json(members["plate.json"], "plate.json")
    seal = _json(members["seal.json"], "seal.json")
    payload = _object(plate.get("payload"), "plate.payload")
    frozen = _object(seal.get("payload"), "seal.payload")
    plate_format = payload.get("format")
    require(plate_format in (PLATE_FORMAT_V1, PLATE_FORMAT_V2)
            and frozen.get("format") == "facet-seal-1",
            "format", "Unsupported plate or seal format")
    transparent = plate_format == PLATE_FORMAT_V2
    expected_channels = CHANNELS_V2 if transparent else CHANNELS_V1
    require(plate.get("state") == "ready" and seal.get("kind") == "seal"
            and seal.get("state") == "frozen", "metadata_state", "Unexpected embedded plate or seal state")
    plate_id = _text(plate.get("id"), "plate.id")
    exposure_id = _text(plate.get("exposure_id"), "plate.exposure_id")
    seal_id = _text(seal.get("id"), "seal.id")
    require(payload.get("seal_id") == seal_id, "seal_id", "Plate refers to another seal")
    content_hash = _checksum(frozen.get("content_hash"), "content_hash")
    require(_digest({key: value for key, value in frozen.items() if key != "content_hash"}) == content_hash,
            "seal_hash", "Frozen canonical content checksum mismatch")
    require(_checksum(payload.get("seal_hash"), "seal_hash") == content_hash,
            "seal_hash", "Plate checksum does not name its frozen content")
    lens = _object(frozen.get("lens"), "lens")
    require(_checksum(payload.get("lens_hash"), "lens_hash") == _digest(lens),
            "lens_hash", "Plate lens checksum mismatch")
    width = _integer(payload.get("width"), 16, MAX_SIDE, "width")
    height = _integer(payload.get("height"), 16, MAX_SIDE, "height")
    request = _object(payload.get("request"), "request")
    require(type(request.get("width")) is int and request["width"] == width
            and type(request.get("height")) is int and request["height"] == height,
            "extent", "Request dimensions disagree with the plate")
    samples = _integer(request.get("samples"), 1, 2, "samples")
    _text(request.get("renderer"), "request.renderer")
    objects = _objects(frozen, payload.get("objects"))
    descriptors = _object(payload.get("passes"), "passes")
    require(set(descriptors) == expected_channels, "channels",
            "Channel set does not match the frozen plate format")
    if transparent:
        _validate_transparency(payload, request)
    for name in sorted(expected_channels):
        descriptor = _object(descriptors[name], "descriptor")
        require(set(descriptor) == {"sha256", "size"}, "descriptor", "Unexpected descriptor fields")
        data = members["passes/" + name]
        size = _integer(descriptor.get("size"), 0, 64*1024*1024, "descriptor.size")
        require(size == len(data), "channel_size", "Channel size mismatch: " + name)
        require(_checksum(descriptor.get("sha256"), "descriptor.sha256") == hashlib.sha256(data).hexdigest(),
                "channel_hash", "Channel checksum mismatch: " + name)
        if not name.endswith(".png"):
            stride = RAW_STRIDE[name]
            require(size == width * height * stride, "extent", "Raw channel extent mismatch: " + name)
    require(_checksum(payload.get("visual_hash"), "visual_hash") == _digest(descriptors),
            "visual_hash", "Channel manifest checksum mismatch")
    decoded = {name: decode_rgb(members["passes/" + name], width, height)
               for name in ("color.png", "depth.png", "object.png")}
    require(decoded["color.png"] == members["passes/color.rgb"],
            "color_pixels", "PNG pixels disagree with the raw color channel")
    depths = struct.iter_unpack("<f", members["passes/depth.f32"])
    indices = struct.iter_unpack("<I", members["passes/object.u32"])
    covered = 0
    background = 0
    minimum = math.inf
    maximum = 0.
    picks = []
    sample_offsets = {0, width-1, (height//2)*width+width//2, width*height-1}
    first_foreground = False
    first_background = False
    object_preview = decoded["object.png"]
    truncation_bytes = members["passes/truncated.u8"] if transparent else None
    layers_bytes = members["passes/layers.u16"] if transparent else None
    contributor_bytes = members["passes/contributor.u32"] if transparent else None
    contributor_png = (decode_rgb(members["passes/contributor.png"], width, height)
                       if transparent else None)
    truncated_pixels = 0
    observed_max = 0
    for offset, ((depth,), (index,)) in enumerate(zip(depths, indices)):
        if index == 0:
            require(depth == math.inf, "depth_background", "Background must have positive infinite depth")
            background += 1
            take = not first_background
            first_background = True
            if transparent:
                require(truncation_bytes[offset] == 0 and
                        struct.unpack_from("<H", layers_bytes, offset*2)[0] == 0 and
                        struct.unpack_from("<I", contributor_bytes, offset*4)[0] == 0,
                        "transparency_background", "Background pixel carries transparency data")
        else:
            require(str(index) in objects, "object_index", "Pixel refers to an unknown object")
            require(math.isfinite(depth) and depth > 0,
                    "depth_value", "Foreground depth must be finite and positive")
            covered += 1
            minimum = min(minimum, depth)
            maximum = max(maximum, depth)
            take = not first_foreground
            first_foreground = True
            if transparent:
                contributor = struct.unpack_from("<I", contributor_bytes, offset*4)[0]
                observed = struct.unpack_from("<H", layers_bytes, offset*2)[0]
                flag = truncation_bytes[offset]
                require(flag in (0, 1), "truncation", "Truncation flag must be binary")
                require(observed >= 1, "layers", "Foreground pixel must observe at least one layer")
                require(contributor == 0 or str(contributor) in objects,
                        "contributor_index", "Contributor refers to an unknown object")
                truncated_pixels += flag
                observed_max = max(observed_max, observed)
                expected_c = bytes(((contributor * factor) % 255) if contributor else 0
                                   for factor in CONTRIBUTOR_FACTORS)
                require(contributor_png[offset*3:offset*3+3] == expected_c,
                        "contributor_preview", "Contributor preview disagrees with its identifier")
        expected_color = bytes(((index * factor) % 255) if index else 0 for factor in OBJECT_FACTORS)
        require(object_preview[offset*3:offset*3+3] == expected_color,
                "object_preview", "Object preview disagrees with the identifier channel")
        if take or offset in sample_offsets:
            pick = {"x": offset % width, "y": offset // width, "object_index": index,
                    "object": objects.get(str(index)), "depth": depth if index else None,
                    "seal_id": seal_id}
            if transparent:
                cid = struct.unpack_from("<I", contributor_bytes, offset*4)[0]
                pick.update({"surface_object": objects.get(str(index)),
                             "contributor_index": cid,
                             "contributor_object": objects.get(str(cid)),
                             "observed_layers": struct.unpack_from("<H", layers_bytes, offset*2)[0],
                             "truncated": bool(truncation_bytes[offset])})
            picks.append(pick)
    statistics = _object(payload.get("statistics"), "statistics")
    require(statistics.get("byte_order") == "little"
            and statistics.get("background_depth") == "positive_infinity"
            and statistics.get("depth_units") == "camera_forward_distance",
            "channel_semantics", "Unexpected raw-channel interpretation")
    require(type(statistics.get("covered_pixels")) is int and statistics["covered_pixels"] == covered,
            "coverage", "Coverage count disagrees with the identifier channel")
    transparency_summary = None
    if transparent:
        require(type(statistics.get("truncated_pixels")) is int and
                statistics["truncated_pixels"] == truncated_pixels,
                "truncation", "Truncation statistics disagree with the truncation channel")
        require(type(statistics.get("observed_layer_max")) is int and
                statistics["observed_layer_max"] == observed_max,
                "layers", "Observed layer maximum disagrees with the layers channel")
        require(statistics["layer_budget"] == payload["transparency"]["layer_budget"],
                "transparency", "Layer budget differs between statistics and frozen semantics")
        transparency_summary = {
            "model": payload["transparency"]["model"],
            "layer_budget": statistics["layer_budget"],
            "observed_layer_max": observed_max,
            "truncated_pixels": truncated_pixels,
            "truncated_samples": statistics["truncated_samples"],
            "overflow_fragments": statistics["overflow_fragments"],
            "identity_lost_beyond_budget": truncated_pixels > 0}
    report = {
        "valid": True, "format": plate_format,
        "summary": {"width": width, "height": height, "samples": samples,
                    "covered_pixels": covered, "background_pixels": background,
                    "objects": len(objects), "finite_depth_range": [minimum, maximum] if covered else None,
                    "depth_units": "camera_forward_distance", "object_id_scope": "local_to_this_plate",
                    "transparency": transparency_summary},
        "provenance": {"plate_id": plate_id, "exposure_id": exposure_id, "seal_id": seal_id,
                       "seal_hash": content_hash, "lens_hash": payload["lens_hash"],
                       "visual_hash": payload["visual_hash"], "renderer": request["renderer"],
                       "assembly_id": _text(frozen.get("assembly_id"), "assembly_id"),
                       "assembly_revision": _integer(frozen.get("assembly_revision"), 1, 2**63-2, "assembly_revision"),
                       "objects": objects,
                       "frozen_resources": [
                           {"key": item["placement"]["key"],
                            "mesh_id": item["placement"]["mesh_id"], "mesh_revision": item["mesh_revision"],
                            "pigment_id": item["placement"]["pigment_id"], "pigment_revision": item["pigment_revision"]}
                           for item in sorted(frozen["closure"], key=lambda item: item["placement"]["key"])]},
        "channels": descriptors, "picks": picks, "trust": dict(TRUST),
    }
    return report, members


def _validate_transparency(payload, request):
    semantics = _object(payload.get("transparency"), "transparency")
    require(semantics.get("model") == "front_to_back_over"
            and semantics.get("surface_identity") == "nearest_named_fragment"
            and semantics.get("contributor_identity") == "largest_named_color_contribution"
            and semantics.get("overflow") == "order_independent_weighted_tail",
            "transparency", "Unsupported frozen transparency semantics")
    budget = _integer(semantics.get("layer_budget"), 1, 64, "transparency.layer_budget")
    require(payload.get("renderer") == "facet-raster-2" and request.get("renderer") == "facet-raster-2",
            "transparency", "Transparent plate must name the transparency renderer")
    statistics = _object(payload.get("statistics"), "statistics")
    require(statistics.get("layer_budget") == budget,
            "transparency", "Statistics and frozen semantics disagree on the layer budget")
