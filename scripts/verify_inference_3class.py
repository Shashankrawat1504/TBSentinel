"""Live end-to-end verification of the three-class TBSentinel inference API.

Unlike the unit tests (which stub the model), this script talks to a *running*
inference service and posts real chest radiographs from the held-out multi-class
test split. It fails loudly if the API does not behave like the three-class
service the dashboard expects.

Usage
-----
Start the API first, then run this script from the project root::

    python -m src.inference.main                 # terminal 1
    python scripts/verify_inference_3class.py    # terminal 2

Environment
-----------
``TBSENTINEL_API_URL``  base URL of the API (default ``http://127.0.0.1:8000``)
``TBSENTINEL_SAMPLES``  images per class to post (default 6)

Exit code is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import base64
import csv
import io
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "multiclass_test.csv"
LABELS = {0: "normal", 1: "tb", 2: "pneumonia"}
CLASS_NAMES = ("normal", "tb", "pneumonia")
PER_CLASS = int(os.environ.get("TBSENTINEL_SAMPLES", "6"))
BASE_URL = os.environ.get("TBSENTINEL_API_URL", "http://127.0.0.1:8000").rstrip("/")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one check and print its outcome."""
    safe = str(detail).encode("ascii", "replace").decode("ascii")
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {safe}")
    if not ok:
        failures.append(name)


def sample_images() -> dict[str, list[str]]:
    """First ``PER_CLASS`` image paths per class from the test split."""
    if not TEST_SPLIT.exists():
        print(f"test split not found: {TEST_SPLIT}")
        sys.exit(1)
    with TEST_SPLIT.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_class: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_class[LABELS[int(row["label"])]].append(row["image_path"])
    return by_class


def main() -> int:
    """Run every API check against the live service."""
    by_class = sample_images()

    # -- /health ------------------------------------------------------------
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=60)
    except requests.exceptions.RequestException as exc:
        print(f"cannot reach {BASE_URL}: {exc}\nIs the API running?")
        return 1

    check(
        "health returns 200",
        health_response.status_code == 200,
        str(health_response.status_code),
    )
    health = health_response.json()
    check(
        "health reports the three-class task",
        health.get("task") == "multiclass_normal_tb_pneumonia",
        str(health.get("task")),
    )
    check(
        "health reports the class names",
        health.get("class_names") == list(CLASS_NAMES),
        str(health.get("class_names")),
    )
    check(
        "health reports num_classes",
        health.get("num_classes") == len(CLASS_NAMES),
        str(health.get("num_classes")),
    )
    check(
        "served checkpoint is the multi-class one",
        "multiclass" in str(health.get("checkpoint_path")),
        str(health.get("checkpoint_path")),
    )

    # -- /predict on real images -------------------------------------------
    total = correct = 0
    for true_class, paths in sorted(by_class.items()):
        for path in paths[:PER_CLASS]:
            file_path = Path(path)
            if not file_path.exists():
                check(f"sample exists ({path})", False, "missing on disk")
                continue
            started = time.perf_counter()
            response = requests.post(
                f"{BASE_URL}/predict",
                files={"file": (file_path.name, file_path.read_bytes(), "image/png")},
                timeout=300,
            )
            latency = time.perf_counter() - started
            if response.status_code != 200:
                check(
                    f"predict 200 ({true_class})",
                    False,
                    f"{response.status_code} {response.text[:160]}",
                )
                continue

            payload = response.json()
            probabilities = payload["probabilities"]
            predicted = payload["predicted_class_name"]
            interpretation = payload.get("interpretation") or {}
            total += 1
            correct += int(predicted == true_class)
            check(
                f"predict contract ({true_class} -> {predicted})",
                set(probabilities) == set(CLASS_NAMES)
                and abs(sum(probabilities.values()) - 1.0) < 1e-6
                and abs(probabilities[predicted] - payload["confidence"]) < 1e-9
                and bool(interpretation.get("headline"))
                and len(interpretation.get("notes") or []) >= 1,
                f"p={ {k: round(v, 4) for k, v in probabilities.items()} } "
                f"latency={latency:.2f}s",
            )

    check("sampled images classified", total >= 3 * PER_CLASS, f"{total} images")
    check(
        "sampled accuracy is sane (> 0.7)",
        correct / max(total, 1) > 0.7,
        f"{correct}/{total}",
    )

    # -- /explain with Grad-CAM --------------------------------------------
    for class_name in CLASS_NAMES:
        paths = by_class.get(class_name) or []
        if not paths:
            continue
        file_path = Path(paths[0])
        response = requests.post(
            f"{BASE_URL}/explain",
            files={"file": (file_path.name, file_path.read_bytes(), "image/png")},
            timeout=300,
        )
        if response.status_code != 200:
            check(
                f"explain 200 ({class_name})",
                False,
                f"{response.status_code} {response.text[:160]}",
            )
            continue
        payload = response.json()
        images = {
            key: Image.open(io.BytesIO(base64.b64decode(payload[key])))
            for key in ("heatmap_base64", "overlay_base64", "rgb_image_base64")
        }
        png_headers = all(
            base64.b64decode(payload[key])[:8] == b"\x89PNG\r\n\x1a\n" for key in images
        )
        colours = images["heatmap_base64"].getcolors(maxcolors=1_000_000) or []
        low, high = images["heatmap_base64"].convert("L").getextrema()
        overlay_differs = (
            images["overlay_base64"].tobytes() != images["rgb_image_base64"].tobytes()
        )
        check(
            f"grad-cam artifacts valid ({class_name})",
            png_headers
            and len(colours) > 50
            and high > low
            and overlay_differs
            and payload.get("target_class_name") == payload.get("predicted_class_name"),
            f"sizes={ {k: v.size for k, v in images.items()} } colours={len(colours)} "
            f"target_class={payload.get('target_class_name')}",
        )

    # -- error handling ----------------------------------------------------
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
        timeout=60,
    )
    check(
        "non-image upload is rejected with 400",
        response.status_code == 400,
        f"{response.status_code} {response.text[:120]}",
    )

    corrupt = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    for endpoint in ("/predict", "/explain"):
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            files={"file": ("broken.png", corrupt, "image/png")},
            timeout=60,
        )
        check(
            f"undecodable upload is a 400 on {endpoint}",
            response.status_code == 400,
            f"{response.status_code} {response.text[:120]}",
        )

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1
    print("ALL API CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
