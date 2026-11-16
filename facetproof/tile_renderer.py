"""Resume high-sample raster windows, then assemble for existing passes."""
from .raster import rasterize
from .transparency import rasterize_transparent, composite
from .tile_protocol import assemble, encode
from .tile_store import TileStore


class TileEngine:
    def __init__(self, db, exposures, blobs, *, tile_size=64):
        self.store = TileStore(db, exposures, blobs)
        self.blobs, self.tile_size = blobs, tile_size

    def _render_window(self, plan, checkpoint, bounds):
        if plan.get("transparent"):
            raster = rasterize_transparent(plan, checkpoint, bounds)
            # Compositing happens per window; the assembled tile already holds
            # final sample channels, matching the opaque engine's contract.
            return composite(plan, raster)
        return rasterize(plan, checkpoint, bounds)

    def render(self, exposure, plan, checkpoint):
        frame, records = self.store.prepare(exposure, plan, self.tile_size)
        transparent = bool(plan.get("transparent"))
        for record in records:
            checkpoint()
            if record["state"] == "complete":
                continue
            # Window bounds stay in the original global sample coordinates.
            # Re-triangulation or shifting float vertices would change edge
            # evaluation and could change f64 depth or coplanar ownership.
            rendered = self._render_window(plan, checkpoint, tuple(record["bounds"]))
            payload = encode(rendered, record["bounds"], frame["fingerprint"], self.blobs,
                             transparent=transparent, objects=plan["objects"])
            checkpoint()
            self.store.commit(exposure, record["ordinal"], payload)
        self.store.assert_owned(exposure)
        frame, records = self.store.snapshot(exposure["id"])
        result = assemble(plan, records, frame["tile_size"], frame["fingerprint"], self.blobs, checkpoint)
        self.store.assert_owned(exposure)
        return result

    def progress(self, exposure_id):
        return self.store.progress(exposure_id)
