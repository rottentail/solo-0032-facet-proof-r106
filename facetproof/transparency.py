"""Order-independent bounded transparency.

Each high-sample pixel keeps an explicit, depth-sorted list of named fragment
layers up to a frozen ``layer_budget``. Visibility between named layers is
decided purely by camera depth, so intersecting shells occlude exactly as their
geometry dictates and reordering placements cannot change a pixel.

Memory is linear in the layer budget rather than in the number of overlapping
triangles. Two storage regions serve two clearly different guarantees:

* Named slots keep per-layer color, alpha, depth and object identity and are
  composited with exact front-to-back Porter-Duff over.
* Once the slots fill, every deeper fragment enters a single order-independent
  overflow tail. The tail keeps the exact aggregate coverage
  ``1 - product(1 - alpha)`` and an alpha-weighted average color in fixed
  memory, so every overflow fragment still contributes color, but its
  individual depth ordering and object identity are deliberately lost. The
  truncation counters make exactly which samples lost detail observable.

Two distinct identities are produced and never conflated:

* surface    - the nearest named fragment (pick the shell to edit its material);
* contributor- the named fragment adding the most color to the final pixel
               (locate the interior part that dominates the view).
"""
from array import array
import math

from .values import require

EPSILON = 1e-9


def _edge(a, b, x, y):
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def _over(dst_rgb, dst_a, src_rgb, src_a):
    """Porter-Duff over: src placed in front of dst, premultiplied math."""
    out_a = src_a + dst_a * (1 - src_a)
    if out_a <= 0:
        return list(dst_rgb), 0.
    channels = []
    for channel in range(3):
        premultiplied = src_rgb[channel] * src_a + dst_rgb[channel] * dst_a * (1 - src_a)
        channels.append(premultiplied / out_a)
    return channels, out_a


def rasterize_transparent(plan, checkpoint, window=None):
    full_width, full_height = plan["width"], plan["height"]
    x0, y0, x1, y1 = (0, 0, full_width, full_height) if window is None else window
    if not (0 <= x0 < x1 <= full_width and 0 <= y0 < y1 <= full_height):
        raise ValueError("Raster window must be a nonempty rectangle inside the canvas")
    width, height = x1 - x0, y1 - y0
    length = width * height
    budget = plan["layer_budget"]
    require(1 <= budget <= 64, "layer_budget", "Layer budget must stay within the frozen bound", 422)

    # Named slots per sample, each stored in structure-of-arrays form.
    slot_depth = array("d", [math.inf]) * (length * budget)
    slot_id = array("I", [0]) * (length * budget)
    slot_face = array("i", [-1]) * (length * budget)
    slot_alpha = array("f", [0.]) * (length * budget)
    slot_color = array("f", [0.]) * (length * budget * 3)
    count = array("H", [0]) * length                 # named layers actually used
    # Order-independent overflow tail (one per sample). No depth is stored, so
    # insertion is commutative and the bound never grows with fragment count.
    tail_weighted = array("f", [0.]) * (length * 3)  # sum of color * alpha
    tail_alpha_weight = array("f", [0.]) * length    # sum of alpha, for averaging
    tail_transmittance = array("f", [1.]) * length   # product of (1 - alpha)
    overflow_fragments = array("I", [0]) * length    # fragments folded into the tail
    fragments = 0

    def add_to_tail(offset, layer_color, layer_alpha):
        tbase = offset * 3
        if layer_alpha <= 0:
            return
        for channel in range(3):
            tail_weighted[tbase + channel] += layer_color[channel] * layer_alpha
        tail_alpha_weight[offset] += layer_alpha
        tail_transmittance[offset] *= (1 - layer_alpha)
        overflow_fragments[offset] += 1

    for triangle in plan["triangles"]:
        checkpoint()
        a, b, c = triangle["vertices"]
        area = _edge(a, b, c[0], c[1])
        if abs(area) < 1e-10:
            continue
        left = max(x0, math.floor(min(a[0], b[0], c[0])))
        right = min(x1 - 1, math.ceil(max(a[0], b[0], c[0])))
        top = max(y0, math.floor(min(a[1], b[1], c[1])))
        bottom = min(y1 - 1, math.ceil(max(a[1], b[1], c[1])))
        obj_id = triangle["object_id"]
        face = triangle["face"]
        alpha = triangle["opacity"]
        color = triangle["color"]
        for y in range(top, bottom + 1):
            if y % 32 == 0:
                checkpoint()
            for x in range(left, right + 1):
                wa = _edge(b, c, x + .5, y + .5) / area
                wb = _edge(c, a, x + .5, y + .5) / area
                wc = 1 - wa - wb
                if min(wa, wb, wc) < -1e-10:
                    continue
                if plan["projection"] == "perspective":
                    distance = 1 / (wa / a[2] + wb / b[2] + wc / c[2])
                else:
                    distance = wa * a[2] + wb * b[2] + wc * c[2]
                # y and x are global sample coordinates; storage is window-local.
                offset = (y - y0) * width + (x - x0)
                used = count[offset]
                base = offset * budget
                # Named slots are filled front-to-back. When full, anything
                # deeper than the farthest named layer belongs to the tail.
                if used < budget:
                    position = used
                    for position in range(used):
                        slot_index = base + position
                        if _closer(distance, obj_id, face,
                                   slot_depth[slot_index], slot_id[slot_index], slot_face[slot_index]):
                            break
                    else:
                        position = used
                    slot_index = base + position
                    # Shift the displaced suffix one slot toward the back.
                    for move in range(used, position, -1):
                        src, dst = base + move - 1, base + move
                        slot_depth[dst] = slot_depth[src]
                        slot_id[dst] = slot_id[src]
                        slot_face[dst] = slot_face[src]
                        slot_alpha[dst] = slot_alpha[src]
                        for channel in range(3):
                            slot_color[dst * 3 + channel] = slot_color[src * 3 + channel]
                    slot_depth[slot_index] = distance
                    slot_id[slot_index] = obj_id
                    slot_face[slot_index] = face
                    slot_alpha[slot_index] = alpha
                    for channel in range(3):
                        slot_color[slot_index * 3 + channel] = color[channel]
                    count[offset] = used + 1
                else:
                    farthest = base + budget - 1
                    if _closer(distance, obj_id, face,
                               slot_depth[farthest], slot_id[farthest], slot_face[farthest]):
                        # The newcomer displaces the previous farthest named
                        # layer into the order-independent tail, then takes the
                        # freed named slot. Tail insertion needs no depth.
                        add_to_tail(offset,
                                    [slot_color[farthest * 3 + ch] for ch in range(3)],
                                    slot_alpha[farthest])
                        position = budget - 1
                        for position in range(budget):
                            slot_index = base + position
                            if _closer(distance, obj_id, face,
                                       slot_depth[slot_index], slot_id[slot_index], slot_face[slot_index]):
                                break
                        else:
                            position = budget - 1
                        for move in range(budget - 1, position, -1):
                            src, dst = base + move - 1, base + move
                            slot_depth[dst] = slot_depth[src]
                            slot_id[dst] = slot_id[src]
                            slot_face[dst] = slot_face[src]
                            slot_alpha[dst] = slot_alpha[src]
                            for channel in range(3):
                                slot_color[dst * 3 + channel] = slot_color[src * 3 + channel]
                        target = base + position
                        slot_depth[target] = distance
                        slot_id[target] = obj_id
                        slot_face[target] = face
                        slot_alpha[target] = alpha
                        for channel in range(3):
                            slot_color[target * 3 + channel] = color[channel]
                    else:
                        add_to_tail(offset, color, alpha)
                fragments += 1
    checkpoint()
    return {"width": full_width, "height": full_height, "window": (x0, y0, x1, y1),
            "budget": budget, "count": count,
            "slot_depth": slot_depth, "slot_id": slot_id, "slot_face": slot_face,
            "slot_alpha": slot_alpha, "slot_color": slot_color,
            "tail_weighted": tail_weighted, "tail_alpha_weight": tail_alpha_weight,
            "tail_transmittance": tail_transmittance, "overflow_fragments": overflow_fragments,
            "fragments": fragments, "triangles": len(plan["triangles"])}


def _closer(distance, obj_id, face, other_distance, other_id, other_face):
    """Explicit, placement-independent depth ordering with stable tie breaks."""
    if distance < other_distance - EPSILON:
        return True
    if distance > other_distance + EPSILON:
        return False
    if obj_id != other_id:
        return obj_id < other_id
    return face < other_face


def _tail_layer(raster, offset):
    """Resolve the order-independent tail into one aggregate over layer."""
    alpha_weight = raster["tail_alpha_weight"][offset]
    if alpha_weight <= 0:
        return None
    coverage = 1 - raster["tail_transmittance"][offset]
    tbase = offset * 3
    color = [raster["tail_weighted"][tbase + channel] / alpha_weight for channel in range(3)]
    return color, min(1., max(0., coverage))


def composite(plan, raster):
    """Resolve the bounded fragment tables into sample channels.

    The fragment tables are window-local when rasterizing a tile, so the
    output arrays share the window extent; assembly later places them back at
    the original global sample coordinates.
    """
    x0, y0, x1, y1 = raster["window"]
    width = x1 - x0
    height = y1 - y0
    length = width * height
    budget = raster["budget"]
    background = plan["background"]
    depth = array("d", [math.inf]) * length
    surface = array("I", [0]) * length
    contributor = array("I", [0]) * length
    colors = array("f", [0.]) * (length * 3)
    score = array("f", [0.]) * length
    observed_layers = array("H", [0]) * length
    truncated = bytearray(length)

    for offset in range(length):
        base = offset * budget
        used = raster["count"][offset]
        # The aggregate tail always sits behind every named layer.
        tail = _tail_layer(raster, offset)
        acc = list(background)
        acc_a = 0.
        if tail is not None:
            acc, acc_a = _over(acc, acc_a, tail[0], tail[1])
        best_id, best_score = 0, 0.
        # Named layers compose back-to-front so the nearest is applied last.
        for layer in range(used - 1, -1, -1):
            slot = base + layer
            layer_color = [raster["slot_color"][slot * 3 + ch] for ch in range(3)]
            layer_alpha = raster["slot_alpha"][slot]
            # A named fragment's contribution is the light it adds after every
            # layer in front of it has attenuated the view.
            surviving = 1 - acc_a
            contribution = layer_alpha * surviving
            weighted = sum(layer_color) / 3 * contribution
            obj_id = raster["slot_id"][slot]
            # Stable tie break on object id keeps the choice deterministic.
            if weighted > best_score + 1e-12 or (abs(weighted - best_score) <= 1e-12
                                                 and 0 < obj_id < best_id and weighted > 0):
                best_score, best_id = weighted, obj_id
            acc, acc_a = _over(acc, acc_a, layer_color, layer_alpha)

        for channel in range(3):
            colors[offset * 3 + channel] = min(1., max(0., acc[channel]))
        if used:
            nearest = base
            depth[offset] = raster["slot_depth"][nearest]
            surface[offset] = raster["slot_id"][nearest]
            contributor[offset] = best_id
            observed_layers[offset] = used
            if raster["overflow_fragments"][offset] > 0:
                truncated[offset] = 1
        else:
            for channel in range(3):
                colors[offset * 3 + channel] = background[channel]
        score[offset] = best_score

    return {"depth": depth, "ids": surface, "contributors": contributor, "colors": colors,
            "contribution": score, "observed_layers": observed_layers,
            "overflow_fragments": raster["overflow_fragments"], "truncated": truncated,
            "fragments": raster["fragments"], "triangles": raster["triangles"],
            "layer_budget": budget}
