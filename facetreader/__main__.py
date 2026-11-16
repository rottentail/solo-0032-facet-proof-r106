"""CLI output uses fixed filenames, never ZIP member paths."""
import argparse
import json
from pathlib import Path
import sys
from .archive import load_file
from .errors import InvalidBundle
from .format import evaluate


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise InvalidBundle("arguments", message)


def main(argv=None):
    try:
        parser = Parser(description="Independently inspect an offline Facet Proof plate bundle")
        parser.add_argument("bundle", help="Path to plate-bundle.zip")
        parser.add_argument("--output", type=Path, help="Write summary, provenance, picks and color.png")
        arguments = parser.parse_args(argv)
        report, members = evaluate(load_file(arguments.bundle))
        if arguments.output is not None:
            try:
                arguments.output.mkdir(parents=True, exist_ok=True)
                for name, value in (("summary.json", {"summary": report["summary"], "trust": report["trust"]}),
                                    ("provenance.json", report["provenance"]), ("picks.json", report["picks"])):
                    (arguments.output / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                                  indent=2, allow_nan=False) + "\n", encoding="utf-8")
                (arguments.output / "color.png").write_bytes(members["passes/color.png"])
                # The two identity previews let an offline reviewer reproduce
                # both frozen choices: surface (material editing) and the
                # dominant contributor (interior localization).
                (arguments.output / "object.png").write_bytes(members["passes/object.png"])
                if "passes/contributor.png" in members:
                    (arguments.output / "contributor.png").write_bytes(members["passes/contributor.png"])
            except OSError as exc:
                raise InvalidBundle("output_write", "Cannot write the output directory") from exc
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except InvalidBundle as exc:
        print(json.dumps({"valid": False, "error": exc.code, "message": exc.message}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
