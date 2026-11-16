import argparse
import json
from pathlib import Path
import tempfile
import shutil
from .application import Application
from .errors import Fault, require
from .routing import route_request, PREFIX

FLOWS = ["mesh-intake", "mesh-retirement", "assembly-edit", "lens-edit", "seal-closure", "exposure-submit", "render-plates", "cancel-exposure", "retry-exposure", "verify-plate", "pixel-pick", "compare-plates", "proof-decision", "quarantine-plate", "recover-lease", "export-bundle", "reserve-capacity", "settle-capacity", "resize-capacity", "tile-progress", "offline-inspect", "backup-closure", "content-inspection", "transparency-shell"]

BOX_FACES = [[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[2,6,7],[2,7,3],
             [1,5,6],[1,6,2],[0,3,7],[0,7,4]]
BOX_VERTICES = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]


def _box(call, name, scale):
    vertices = [[coordinate * scale for coordinate in point] for point in BOX_VERTICES]
    return call("POST", "meshes", {"name": name, "vertices": vertices, "faces": BOX_FACES})


def _transparency_flow(call, rejected):
    # A transparent outer shell and a transparent middle shell surround an
    # opaque core, so a center pixel crosses several depth-sorted layers.
    shell = _box(call, "透明外壳", 1.0)
    middle = _box(call, "透明夹层", 0.7)
    core = _box(call, "不透明内芯", 0.35)
    shell_pigment = call("POST", "pigments", {"name": "外壳玻璃", "color": [.1, .3, .9],
                                               "opacity": .25, "two_sided": True})
    middle_pigment = call("POST", "pigments", {"name": "夹层玻璃", "color": [.2, .8, .5],
                                               "opacity": .3, "two_sided": True})
    core_pigment = call("POST", "pigments", {"name": "内芯漆面", "color": [.95, .95, .95],
                                              "opacity": .9, "two_sided": True})
    placements = [
        {"key": "shell", "mesh_id": shell["id"], "pigment_id": shell_pigment["id"]},
        {"key": "middle", "mesh_id": middle["id"], "pigment_id": middle_pigment["id"]},
        {"key": "core", "mesh_id": core["id"], "pigment_id": core_pigment["id"]}]

    def render(order, key, budget):
        assembly = call("POST", "assemblies", {"name": "透明结构", "lens": {"eye": [0, 0, 6]}})
        call("PUT", "assemblies/"+assembly["id"], {"revision": 1, "placements": order})
        seal = call("POST", "assemblies/"+assembly["id"]+"/seal", {"revision": 2})
        call("POST", "seals/"+seal["id"]+"/exposures",
             {"key": key, "width": 64, "height": 64, "samples": 1, "layer_budget": budget})
        rendered = call("POST", "worker/step")
        require(rendered["state"] == "succeeded", "commission", str(rendered), 500)
        return seal, rendered["plate"]

    seal, plate = render(placements, "transparent-reference", 8)
    payload = plate["payload"]
    require(payload["format"] == "facet-plate-2", "commission", "Transparent scene must use plate v2", 500)
    stats = payload["statistics"]
    require(stats["transparency"]["model"] == "front_to_back_over"
            and stats["layer_budget"] == 8, "commission", "Frozen transparency semantics missing", 500)
    # The two identity uses are separate channels and must not be conflated.
    pick = call("GET", "plates/"+plate["id"]+"/pick?x=32&y=32")
    require(pick["surface_object"]["key"] == "shell", "commission", "Surface identity is not the outer shell", 500)
    require(pick["contributor_object"]["key"] == "core", "commission", "Contributor identity is not the inner core", 500)
    require(pick["semantics"] == "facet-transparency-1", "commission", "Pick lacks frozen semantics", 500)

    # Reversing part order cannot change any channel: depth, not submission
    # order, decides occlusion.
    _, reversed_plate = render(list(reversed(placements)), "transparent-reversed", 8)
    require(reversed_plate["payload"]["visual_hash"] == payload["visual_hash"],
            "commission", "Reordering placements changed the image", 500)

    # A budget below the overlap count truncates identity but preserves color;
    # the statistics say exactly where detail was lost.
    _, tight_plate = render(placements, "transparent-tight", 1)
    tight = tight_plate["payload"]["statistics"]
    require(tight["truncated_pixels"] > 0 and tight["overflow_fragments"] > 0,
            "commission", "Overflow was not reported under a tight layer budget", 500)
    tight_pick = call("GET", "plates/"+tight_plate["id"]+"/pick?x=32&y=32")
    require(tight_pick["truncated"] and tight_pick["identity_loss"] == "overflow_tail",
            "commission", "Truncated pixel was not flagged for review", 500)
    color_a = call("GET", "plates/"+plate["id"]+"/passes/color.rgb")[0]
    color_b = call("GET", "plates/"+tight_plate["id"]+"/passes/color.rgb")[0]
    # No overflow fragment is discarded: the tail retains every fragment's
    # color via alpha-weighted aggregation and exact aggregate coverage. A
    # tighter budget moves more layers from exact over into the documented
    # order-independent tail approximation, so colors stay close rather than
    # bit-identical. Bound that approximation and report the observed delta.
    max_color_delta = max(abs(left - right) for left, right in zip(color_a, color_b))
    require(max_color_delta <= 48, "commission",
            "Tail approximation exceeded its bounded color deviation", 500)

    # The independent offline reader reconstructs the same frozen semantics.
    from facetreader import validate_bundle
    report = validate_bundle(call("GET", "plates/"+plate["id"]+"/bundle")[0])
    require(report["valid"] and report["format"] == "facet-plate-2"
            and report["summary"]["transparency"]["model"] == "front_to_back_over",
            "commission", str(report), 500)
    result = {"workflow": "transparency-shell", "statistics": stats,
              "pick": {"surface": pick["surface_object"]["key"],
                       "contributor": pick["contributor_object"]["key"],
                       "observed_layers": pick["observed_layers"]},
              "order_invariant": True,
              "truncation": {"truncated_pixels": tight["truncated_pixels"],
                             "overflow_fragments": tight["overflow_fragments"],
                             "no_fragment_dropped": True,
                             "tail_approximation": "order_independent_weighted",
                             "max_color_byte_delta": max_color_delta},
              "offline": report["summary"]["transparency"],
              "rejected": rejected("POST", "seals/"+seal["id"]+"/exposures",
                                   {"key": "bad-budget", "width": 32, "height": 32,
                                    "layer_budget": 0}, "integer")}
    return result


def commission(root, flow, output=None):
    app = Application(root)
    def call(method, route, data=None): return route_request(app, method, PREFIX + route, data)
    def rejected(method, route, data, code):
        try: call(method, route, data)
        except Fault as exc:
            require(exc.code == code, "commission", f"Expected {code}, got {exc.code}", 500)
            return exc.code
        raise Fault("commission", "Expected rejection", 500)
    mesh_data = {"name":"斜翼", "vertices":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[0,0,1.2]], "faces":[[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,3,2],[0,2,1]]}
    mesh = call("POST", "meshes", mesh_data)
    pigment = call("POST", "pigments", {"name":"釉青", "color":[.12,.62,.61], "two_sided":True})
    assembly = call("POST", "assemblies", {"name":"斜翼结构说明图", "lens":{"eye":[3,2,4]}})
    assembly = call("PUT", "assemblies/"+assembly["id"], {"revision":1, "placements":[{"key":"wing", "mesh_id":mesh["id"], "pigment_id":pigment["id"], "transform":{"euler":[0,20,0]}}]})
    result = {"workflow":flow}
    if flow == "transparency-shell":
        return _transparency_flow(call, rejected)
    if flow == "mesh-intake":
        result["rejected"] = rejected("POST", "meshes", {**mesh_data, "faces":[[0,0,1]]}, "face")
        return result
    if flow == "mesh-retirement":
        call("POST", "meshes/"+mesh["id"]+"/retire", {"revision":1})
        result["rejected"] = rejected("POST", "assemblies/"+assembly["id"]+"/seal", {"revision":2}, "retired")
        return result
    if flow in ("assembly-edit", "lens-edit"):
        changes = {"name":"更新说明图"} if flow == "assembly-edit" else {"lens":{"projection":"orthographic"}}
        edited = call("PUT", "assemblies/"+assembly["id"], {"revision":2, **changes})
        result.update(revision=edited["revision"], rejected=rejected("PUT", "assemblies/"+assembly["id"], {"revision":2, "name":"过期"}, "stale_revision"))
        return result
    seal = call("POST", "assemblies/"+assembly["id"]+"/seal", {"revision":2})
    if flow == "seal-closure":
        call("PUT", "pigments/"+pigment["id"], {"revision":1, "name":"改色", "color":[1,0,0]})
        require(call("GET", "seals/"+seal["id"]) == seal, "commission", "Seal changed", 500)
        result["rejected"] = rejected("PUT", "assemblies/"+assembly["id"], {"revision":3, "name":"改稿"}, "sealed")
        return result
    request = {"key":"commission", "width":96, "height":96, "samples":2}
    if flow in {'reserve-capacity', 'settle-capacity', 'resize-capacity'}:
        request.update(width=16, height=16, samples=1)
        call('PUT', 'resources', {'capacity':256, 'revision':1})
        exposure = call('POST', 'seals/'+seal['id']+'/exposures', request)
        result['reserved'] = call('GET', 'resources')['reserved']
        require(result['reserved'] == 256, 'commission', 'Reservation missing', 500)
        result['rejected'] = rejected('POST', 'seals/'+seal['id']+'/exposures', {**request,'key':'too-many'}, 'capacity_reserved')
        if flow == 'reserve-capacity':
            call('POST', 'exposures/'+exposure['id']+'/cancel')
        elif flow == 'settle-capacity':
            claim = app.exposures.claim()
            call('POST', 'exposures/'+exposure['id']+'/cancel')
            require(call('GET', 'resources')['reserved'] == 256, 'commission', 'Accepted cancellation released running work', 500)
            app.exposures.fail(exposure['id'], claim['token'], 'cancel_checkpoint')
        else:
            call('PUT', 'resources', {'capacity':512, 'revision':2})
            second = call('POST', 'seals/'+seal['id']+'/exposures', {**request,'key':'second'})
            policy = call('PUT', 'resources', {'capacity':256, 'revision':3})
            require(policy['overcommitted'] == 256, 'commission', 'Prior promises were erased', 500)
            rejected('PUT', 'resources', {'capacity':512,'revision':3}, 'stale_revision')
            require(call('POST', 'worker/step')['state']=='succeeded','commission','Promised work lost',500)
            require(call('POST', 'worker/step')['state']=='succeeded','commission','Second promise lost',500)
            result['settled'] = second['id']
        require(call('GET', 'resources')['reserved'] == 0, 'commission', 'Settled work leaked capacity', 500)
        result['capacity'] = call('GET', 'resources')
        return result
    exposure = call("POST", "seals/"+seal["id"]+"/exposures", request)
    eroute = "exposures/"+exposure["id"]
    if flow == 'tile-progress':
        require(not call('GET', eroute+'/progress')['planned'], 'commission', 'Queued work claims completed pixels', 500)
    if flow == "exposure-submit":
        require(call("POST", "seals/"+seal["id"]+"/exposures", request)["id"] == exposure["id"], "commission", "Idempotency mismatch", 500)
        result["rejected"] = rejected("POST", "seals/"+seal["id"]+"/exposures", {**request, "width":128}, "key_conflict")
        return result
    if flow == "cancel-exposure":
        result["state"] = call("POST", eroute+"/cancel")["state"]
        require(call("POST", "worker/step")["state"] == "idle", "commission", "Cancelled work claimed", 500)
        result["rejected"] = rejected("POST", eroute+"/retry", {}, "state")
        return result
    if flow in ("retry-exposure", "recover-lease"):
        claimed = app.exposures.claim(lease_seconds=-1 if flow == "recover-lease" else 120)
        if flow == "retry-exposure":
            app.exposures.fail(exposure["id"], claimed["token"], "interrupted")
            call("POST", eroute+"/retry")
        else:
            require(exposure["id"] in call("POST", "worker/recover")["recovered"], "commission", "Recovery failed", 500)
    rendered = call("POST", "worker/step")
    require(rendered["state"] == "succeeded", "commission", str(rendered), 500)
    plate = rendered["plate"]
    route = "plates/"+plate["id"]
    require(plate["payload"]["statistics"]["covered_pixels"] > 0, "commission", "Empty image", 500)
    result["statistics"] = plate["payload"]["statistics"]
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        for name in ("color.png", "depth.png", "object.png"): (Path(output)/name).write_bytes(call("GET", route+"/passes/"+name)[0])
    if flow == 'offline-inspect':
        from facetreader import validate_bundle
        bundle = call('GET', route+'/bundle')[0]
        report = validate_bundle(bundle)
        require(report['valid'], 'commission', str(report), 500)
        damaged = validate_bundle(bundle[:-20])
        require(not damaged['valid'], 'commission', 'Truncated package was accepted', 500)
        result['offline'] = report['summary']
        result['rejected'] = damaged['error']
    elif flow == 'backup-closure':
        proof = call('POST', route+'/proofs', {'note':'备份前核验'})
        call('POST', 'proofs/'+proof['id']+'/decide', {'state':'accepted','note':'通道完整'})
        backup = call('POST', 'maintenance/backups')
        op = backup['operation_id']
        require(call('GET', 'maintenance/backups/'+op)['state']=='running','commission','Missing running backup',500)
        rejected('GET', 'maintenance/backups/'+op+'/verify', {}, 'backup_incomplete')
        completed = call('POST', 'maintenance/backups/'+op+'/resume')
        require(completed['state']=='succeeded','commission','Backup was not published',500)
        call('POST', route+'/quarantine', {'reason':'备份时点后的撤回'})
        require(call('GET', 'maintenance/backups/'+op+'/verify')['valid'],'commission','Immutable backup verification failed',500)
        restore_root = Path(root)/'restore-check'
        shutil.copytree(completed['data_directory'], restore_root)
        restored = Application(restore_root)
        require(restored.plates.verify(plate['id'])['valid'] and restored.proofs.get(proof['id'])['state']=='accepted',
                'commission','Restored snapshot mixed later review facts',500)
        require(restored.bundles.export(plate['id']), 'commission', 'Restored export failed', 500)
        result['backup'] = {'state':completed['state'],'references':len(completed['manifest']['closure']['references'])}
        result['rejected'] = rejected('GET', route+'/bundle', {}, 'plate_state')
    elif flow == 'content-inspection':
        result['inspection'] = call('GET', 'maintenance/inspect')
        result['rejected'] = rejected('GET', 'maintenance/backups/unknown', {}, 'backup_id')
    elif flow == 'tile-progress':
        progress = call('GET', eroute+'/progress')
        require(progress['complete'] and progress['committed_tiles'] == 9 and progress['state'] == 'succeeded',
                'commission', 'Persisted progress does not match the completed exposure', 500)
        result['progress'] = progress
        result['verification'] = call('POST', route+'/verify')
    elif flow == "verify-plate": result["verification"] = call("POST", route+"/verify")
    elif flow == "pixel-pick":
        result["pick"] = call("GET", route+"/pick?x=48&y=48")
        require(result["pick"]["object"] is not None, "commission", "Center object missing", 500)
    elif flow == "compare-plates":
        call("POST", "seals/"+seal["id"]+"/exposures", {**request, "key":"second"})
        second = call("POST", "worker/step")["plate"]
        result["comparison"] = call("POST", "comparisons", {"left":plate["id"], "right":second["id"]})["payload"]
        require(result["comparison"]["color_changed"] == 0, "commission", "Determinism failed", 500)
    elif flow in ("proof-decision", "quarantine-plate"):
        proof = call("POST", route+"/proofs", {"note":"检查结构轮廓"})
        result["proof"] = call("POST", "proofs/"+proof["id"]+"/decide", {"state":"accepted", "note":"通过"})["state"]
        if flow == "quarantine-plate":
            call("POST", route+"/quarantine", {"reason":"撤回"})
            require(call("GET", "proofs/"+proof["id"])["state"] == "invalidated", "commission", "Proof remains accepted", 500)
            call("POST", route+"/purge")
            result["rejected"] = rejected("GET", route+"/bundle", {}, "plate_state")
    elif flow == "export-bundle":
        bundle = call("GET", route+"/bundle")[0]
        require(bundle == call("GET", route+"/bundle")[0], "commission", "Bundle changed", 500)
        result["bundle_bytes"] = len(bundle)
    if "rejected" not in result:
        result["rejected"] = rejected("POST", eroute+"/cancel", {}, "state")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS, default="render-plates")
    parser.add_argument("--output")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="facetproof-") as root:
        print(json.dumps(commission(root, args.flow, args.output), ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
