"""Extract evaluation-protocol text from a user-supplied CONSAC paper PDF."""
import argparse
from pathlib import Path

import fitz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the paper PDF")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/consac_protocol_text.txt"),
        help="Destination text file (default: outputs/consac_protocol_text.txt)",
    )
    args = parser.parse_args()

    doc = fitz.open(args.pdf)
    chunks = [f"pages: {len(doc)}"]
    for i, page in enumerate(doc):
        text = page.get_text()
        if any(key in text.lower() for key in ("adelaide", "sequential ransac", "misclassification")):
            chunks.append(f"\n===== page {i + 1} =====\n{text}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(chunks), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
