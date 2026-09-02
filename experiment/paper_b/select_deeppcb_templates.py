from __future__ import annotations

import argparse
from pathlib import Path


def select_templates(pcb_data: Path, split_names: list[str], expected: int) -> list[Path]:
    images = []
    for split_name in split_names:
        split_file = pcb_data / split_name
        for raw in split_file.read_text(encoding="utf-8-sig").splitlines():
            fields = raw.split()
            if not fields:
                continue
            target = Path(fields[0])
            template = pcb_data / target.parent / f"{target.stem}_temp.jpg"
            if not template.is_file():
                raise FileNotFoundError(template)
            images.append(template.resolve())
    unique = sorted(dict.fromkeys(images))
    if len(unique) != expected:
        raise ValueError(f"Official split lists resolve to {len(unique)} templates, expected {expected}")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve DeepPCB templates from official split lists")
    parser.add_argument("--pcb-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-files", nargs="+", default=["trainval.txt", "test.txt"])
    parser.add_argument("--expected", type=int, default=1500)
    args = parser.parse_args()
    images = select_templates(args.pcb_data.resolve(), args.split_files, args.expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{path.as_posix()}\n" for path in images), encoding="utf-8")
    print(f"selected {len(images)} official templates from {args.split_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
