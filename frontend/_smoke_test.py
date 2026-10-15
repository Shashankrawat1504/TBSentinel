"""Smoke test for the new frontend tbui package (storage + metrics + api)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from tbui import api, metrics, storage, theme  # noqa: E402

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        failures.append(name)


# --- storage ---------------------------------------------------------------
storage.init_db()
pid = storage.generate_patient_id()
check("generate_patient_id", pid.startswith("PT-"), pid)

stats = storage.get_stats()
check("get_stats keys", {"total", "normal", "tb", "pneumonia", "non_tb",
                         "patients", "avg_confidence", "tb_rate",
                         "last_at"} <= set(stats), str(stats))

records = storage.list_records()
check("list_records returns list", isinstance(records, list), f"{len(records)} rows")
check("list_records excludes blobs",
      all("image_jpeg" not in r for r in records))

# Round-trip an assessment with a synthetic image + gradcam payload.
import base64  # noqa: E402
import io  # noqa: E402

from PIL import Image  # noqa: E402

buf = io.BytesIO()
Image.new("L", (64, 64), color=128).save(buf, format="PNG")
image_bytes = buf.getvalue()

overlay_b64 = base64.b64encode(image_bytes).decode()
record = {
    "prediction": {
        "predicted_class_name": "pneumonia",
        "confidence": 0.9123,
        "probabilities": {"normal": 0.0211, "tb": 0.0666,
                          "pneumonia": 0.9123},
        "interpretation": {"label": "Pneumonia", "tone": "warning",
                           "headline": "Model screen: pneumonia",
                           "notes": ["smoke test note"]},
    },
    "explanation": {"overlay_base64": overlay_b64, "heatmap_base64": overlay_b64},
    "image_bytes": image_bytes,
    "filename": "test.png",
    "strategy": "Centralized",
    "timestamp": "2026-10-10 12:00:00 UTC",
    "latency_seconds": 0.42,
    "patient_id": pid,
    "patient_name": "Test Patient",
    "age": 40,
    "gender": "Female",
    "notes": "smoke test",
}
record_id = storage.insert_assessment(record)
check("insert_assessment returns id", record_id.startswith("TSR-"), record_id)

full = storage.get_record(record_id)
check("get_record finds row", full is not None)
check("get_record has blobs", bool(full and full["image_jpeg"] and full["overlay_png"]))
check("get_record patient name", full and full["patient_name"] == "Test Patient")

found = storage.list_records(search=pid)
check("search by patient id", any(r["record_id"] == record_id for r in found))

tb_only = storage.list_records(prediction_filter="TB")
check("filter TB", all(r["predicted_class"] == "tb" for r in tb_only))
pneu_only = storage.list_records(prediction_filter="Pneumonia")
check("filter Pneumonia",
      any(r["record_id"] == record_id for r in pneu_only)
      and all(r["predicted_class"] == "pneumonia" for r in pneu_only))
norm_only = storage.list_records(prediction_filter="Normal")
check("filter Normal", all(r["predicted_class"] == "normal"
                           for r in norm_only))
ntb_only = storage.list_records(prediction_filter="Non-TB")
check("filter Non-TB", all(r["predicted_class"] == "non_tb" for r in ntb_only))
check("3-class probabilities stored",
      abs(float(full["prob_normal"]) - 0.0211) < 1e-9
      and abs(float(full["prob_pneumonia"]) - 0.9123) < 1e-9
      and full["prob_non_tb"] is None)
check("interpretation stored", "pneumonia" in (full["interpretation"] or ""))

# A reviewer can point the Grad-CAM card at another class; the record must keep
# the prediction's own explanation (band, focus, notes) rather than borrow that
# class's focus.
cross = dict(record)
cross["prediction_explanation"] = {
    "interpretation": {
        "label": "Pneumonia",
        "tone": "warning",
        "headline": "Model screen: pneumonia (prediction map)",
        "focus": {"available": True, "summary": "prediction map focus"},
    }
}
cross["explanation"] = {
    "overlay_base64": overlay_b64,
    "heatmap_base64": overlay_b64,
    "interpretation": {
        "label": "Tuberculosis",
        "tone": "error",
        "headline": "map recomputed for another class",
        "focus": {"available": True, "summary": "another class focus"},
    },
}
cross_id = storage.insert_assessment(cross)
cross_full = storage.get_record(cross_id)
stored_explanation = cross_full["interpretation"] or ""
check("stored explanation keeps the prediction's own focus",
      "prediction map focus" in stored_explanation
      and "another class focus" not in stored_explanation)
storage.delete_record(cross_id)

stats2 = storage.get_stats()
check("stats increment", stats2["total"] == stats["total"] + 1)

deleted = storage.delete_record(record_id)
check("delete_record", deleted is True)
check("record gone", storage.get_record(record_id) is None)

# Sequential ids must not collide after deletion.
record_id2 = storage.insert_assessment(record)
record_id3 = storage.insert_assessment(record)
check("unique sequential ids", record_id2 != record_id3, f"{record_id2} vs {record_id3}")
storage.delete_record(record_id2)
storage.delete_record(record_id3)

# --- metrics ---------------------------------------------------------------
rows = metrics.load_comparison()
check("3 metric rows", len(rows) == 3)
keys = [r["key"] for r in rows]
check("row keys", keys == ["centralized", "fedavg", "fedprox"], str(keys))
check("accuracy loaded", abs(rows[0]["accuracy"] - 0.9878) < 1e-6)
fedavg = next(r for r in rows if r["key"] == "fedavg")
check("fedavg read from artifacts", "artifacts/outputs" in fedavg["source"], fedavg["source"])
rows = metrics.delta_vs_baseline(rows)
check("delta computed", abs(rows[1]["delta_accuracy"] - (0.9811 - 0.9878)) < 1e-3)

mc = metrics.load_multiclass()
check("multiclass metrics loaded", mc is not None)
if mc:
    check("multiclass classes", mc["class_names"] == ["normal", "tb", "pneumonia"],
          str(mc["class_names"]))
    check("multiclass per-class keys",
          set(mc["per_class"]) == {"normal", "tb", "pneumonia"}
          and all({"precision", "recall", "f1", "auc", "support"}
                  <= set(stats) for stats in mc["per_class"].values()))
    check("multiclass supports sum to samples",
          sum(s["support"] for s in mc["per_class"].values()) == mc["samples"])
    check("multiclass confusion is square",
          len(mc["confusion_matrix"]) == 3
          and all(len(row) == 3 for row in mc["confusion_matrix"]))
    check("multiclass accuracy in range", 0.5 <= mc["accuracy"] <= 1.0,
          f"{mc['accuracy']:.4f}")

# --- api -------------------------------------------------------------------
check("decode_base64_png", api.decode_base64_png(overlay_b64) == image_bytes)
check("strategy centralized", api.model_strategy_label("x/tb_efficientnet_b3_best.pt") == "Centralized")
check("strategy fedavg", api.model_strategy_label("x/tb_federated_fedavg_best.pt") == "Federated (FedAvg)")
check("strategy fedprox", api.model_strategy_label("x/tb_federated_fedprox_best.pt") == "Federated (FedProx)")
check("prediction_interpretation prefers the prediction snapshot",
      api.prediction_interpretation({
          "prediction": {"interpretation": {"headline": "score-level"}},
          "prediction_explanation": {"interpretation": {"headline": "map-level"}},
          "explanation": {"interpretation": {"headline": "another class"}},
      })["headline"] == "map-level")
check("prediction_interpretation rejects another class's map",
      api.prediction_interpretation({
          "prediction": {"predicted_class_name": "pneumonia",
                         "interpretation": {"headline": "score-level"}},
          "explanation": {"target_class_name": "tb",
                          "interpretation": {"headline": "another class"}},
      })["headline"] == "score-level")
check("prediction_interpretation uses the explanation for the predicted class",
      api.prediction_interpretation({
          "prediction": {"predicted_class_name": "pneumonia",
                         "interpretation": {"headline": "score-level"}},
          "explanation": {"target_class_name": "pneumonia",
                          "interpretation": {"headline": "map-level"}},
      })["headline"] == "map-level")
check("prediction_interpretation tolerates an empty record",
      api.prediction_interpretation(None) == {})
hs = api.health_summary(None)
check("health_summary offline", hs["api_online"] is False and hs["model_loaded"] is False)
hs = api.health_summary({"status": "healthy", "model_loaded": True,
                         "checkpoint_path": "C:/x/tb_federated_fedavg_best.pt",
                         "device": "cuda"})
check("health_summary online", hs["api_online"] and hs["model_loaded"] and hs["strategy"] == "Federated (FedAvg)")

# --- theme -----------------------------------------------------------------
assert "ts-stat" in theme.STYLES
html = theme.stat_card("Label", "42", "x", foot="foot", tone="danger")
check("stat_card html", "ts-stat--danger" in html and "42" in html)
check("kv_row escapes", "&lt;b&gt;" in theme.kv_row("k", "<b>"))
check("pill html", "ts-pill--ok" in theme.pill("ok", "ready"))
check("class labels", theme.class_label("tb") == "Tuberculosis"
      and theme.class_label("pneumonia") == "Pneumonia"
      and theme.class_label("normal") == "Normal", theme.class_label("tb"))
check("class badges", theme.class_badge("tb") == ("⚠", "Tuberculosis")
      and theme.badge("pneumonia").count("ts-badge--pneumonia") == 1
      and theme.badge("normal").count("ts-badge--normal") == 1)
check("badge colours mapped", theme.CLASS_TONES == {"normal": "success",
      "tb": "error", "pneumonia": "warning", "non_tb": "success"},
      str(theme.CLASS_TONES))
check("ordered_classes clinical order",
      theme.ordered_classes(["pneumonia", "tb", "normal"])
      == ["normal", "tb", "pneumonia"])
check("unknown class is idle, not a crash",
      theme.class_kind("covid") == "idle" and "ts-badge--idle" in theme.badge("covid"))
check("level_chip html", "High confidence" in theme.level_chip("High", "success")
      and "ts-" not in theme.level_chip("Low", "error").split("font-size")[0])
check("prob_row tones", "ts-bar-fill--tb" in theme.prob_row("TB", 0.9, "tb")
      and "ts-bar-fill--pneumonia" in theme.prob_row("Pneu", 0.5, "pneumonia")
      and "ts-bar-fill--primary" in theme.prob_row("X", 0.5, "unknown-kind"))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL TESTS PASSED")
