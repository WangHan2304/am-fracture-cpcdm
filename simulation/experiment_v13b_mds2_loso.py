# -*- coding: utf-8 -*-
"""R13b - re-run the LOSO block with NIST mds2-3402 added as a third Ti64 source.

Scope is deliberately narrow: this script touches the LEAVE-ONE-SOURCE block only. LOMO and
LOPO are not recomputed, and the locked R13 result file is never written to. Its LOSO block is
read back verbatim and republished under `loso_v13_original` so the old and new statistics sit
side by side in one JSON.

It reuses the R13 machinery by importing `experiment_v13_cv_variants` (its module-level work is
constants only; main() is behind a __main__ guard), so `build_loso`, `decide_s0`, `calibrate`,
`score_points`, `finish_fold` and `summarise` are the *same* code that produced the locked
numbers, and every run is served from the same `cv_prediction_cache.json`.

Output: simulation/output/experiment_v13b_mds2_loso_results.json
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time

SIM = r"D:\20260618断裂模型论文\simulation"
if SIM not in sys.path:
    sys.path.insert(0, SIM)

import experiment_v13_cv_variants as cv          # noqa: E402

OUTPATH = os.path.join(SIM, "output", "experiment_v13b_mds2_loso_results.json")
LOGPATH = os.path.join(SIM, "output", "cv_variants_r13b_log.txt")
NEW_SRC = "nist_mds2_3402"

MECH_NOTE_NIST = (
    "Heat-treatment state and property-range mismatch. The held source spans 15 conditions: HT0 "
    "is as-built, HT1 is a stress-relief anneal, and HT2-HT14 are HIP treatments (14 of them, at "
    "800/920/1050 C, 100/200 MPa, with cooling or quenching from 12 to 2000 C/min, some with a "
    "tempering or ageing step). LOSO calibrates S0 on the AS-BUILT points of the other sources "
    "only (v12_original + sun2024adem), so 28 of the 30 held rows are a material state the "
    "calibration set does not contain. A second, independent gap applies to ALL 30 rows: this "
    "dataset's properties are far outside the calibration range even as-built - mds2 HT0 is UTS "
    "1294-1303 MPa with TE 16.8-19.3 %, against UTS 1050-1299 MPa with TE 4.8-9.3 % for the 6 "
    "as-built Ti64 calibration points. HIP then raises TE further, to 27.3-32.8 % in 13 of the 14 "
    "treated conditions. The attribution between these two gaps is not assumed here; it is "
    "measured by the diagnostic block below and stated in the mechanism sentence appended to this "
    "note."
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel TaylorCPCDM workers (8 was used for the locked R13 run)")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    log = cv.Log(LOGPATH, echo=True)
    t0 = time.time()
    log("=" * 78)
    log("R13b  LOSO re-run with NIST mds2-3402 as a third Ti64 source")
    log("n_grains=%d  S0_grid=%s  workers=%d" % (cv.N_GRAINS, cv.S0_GRID, args.workers))

    locked = json.load(open(cv.RESULT_PATH, encoding="utf-8"))
    locked_loso = locked["leave_one_source"]
    log("locked R13 result : %d LOSO folds, %d held predictions, mean ef %.2f%%"
        % (len(locked_loso["folds"]), locked_loso["summary"]["n_predictions"],
           locked_loso["summary"]["mean_ef_error_pct"]))
    log("locked R13 dataset: %s" % json.dumps(locked["dataset_summary"]["n_by_source"]))

    points = cv.load_extended_dataset()
    by_id = {p["point_id"]: p for p in points}
    log("R13b dataset      : %d points, %d sources"
        % (len(points), len({p["source_id"] for p in points})))

    cache = cv.load_cache()
    log("cache             : %d entries" % len(cache))

    loso_specs, loso_skipped = cv.build_loso(points)
    log("LOSO folds from build_loso: %d (+%d skip note(s))"
        % (len(loso_specs), len(loso_skipped)))
    for sp in loso_specs:
        log("   %-9s held=%-22s train=%-40s n_train=%2d n_held=%2d prefer_locked=%s"
            % (sp["material"], sp["held_source"], ",".join(sp["train_sources"]),
               len(sp["train_points"]), len(sp["held_points"]), sp["prefer_locked"]))
    log("(for comparison, the locked R13 run had %d folds: %s)"
        % (len(locked_loso["folds"]),
           ", ".join("%s/%s" % (f["material"], f["held_source"])
                     for f in locked_loso["folds"])))

    wave1 = cv.grid_task_universe(points)
    log("wave1 grid universe: %d (orientation set is unchanged by mds2, so this is all cached)"
        % len(wave1))
    cv.ensure_runs(wave1, cache, log, workers=args.workers)

    if args.plan_only:
        log("PLAN: would solve S0 for %d folds and score %d held points"
            % (len(loso_specs), sum(len(s["held_points"]) for s in loso_specs)))
        log.close()
        return

    # ---- choose each fold's S0 on its training set only (leakage-free) ----
    for sp in loso_specs:
        sp["S0"], sp["train_err"], sp["method"], sp["trace"] = cv.decide_s0(
            cache, sp["material"], sp["train_points"], sp["prefer_locked"],
            log=log, workers=args.workers)
        log("   -> %-9s held=%-22s S0=%-10.6g method=%-16s train_err=%.2f%%"
            % (sp["material"], sp["held_source"], sp["S0"], sp["method"], sp["train_err"]))

    # ---- wave 2: held-point predictions at the fold S0 and at the locked alt S0 ----
    wave2 = []
    for sp in loso_specs:
        for p in sp["held_points"]:
            wave2.append((sp["material"], p["orientation_deg"], sp["S0"]))
    for sp in loso_specs:
        alt = cv.S0_KNOWN_V12.get(sp["material"])
        if alt is not None and abs(alt - sp["S0"]) > 1e-9:
            for p in sp["held_points"]:
                wave2.append((sp["material"], p["orientation_deg"], alt))
    log("wave2: %d prediction runs requested" % len(wave2))
    n_new, n_fail = cv.ensure_runs(wave2, cache, log, workers=args.workers)
    cv.save_cache(cache, meta={"n_grains": cv.N_GRAINS, "S0_grid": cv.S0_GRID,
                               "note": "R13b appended mds2-3402 fold runs to the R13 cache"})

    # ---- assemble the new LOSO block, using the same finish_fold as R13 ----
    folds = []
    for sp in loso_specs:
        meta = {
            "cv_type": "leave_one_source",
            "held_source": sp["held_source"],
            "train_sources": sp["train_sources"],
            "n_train": len(sp["train_points"]),
            "train_point_ids": [p["point_id"] for p in sp["train_points"]],
            "held_trend": cv.trend_of(sp["held_points"]),
            "held_heat_treatments": sp["held_heat_treatments"],
            "mechanism_note": (MECH_NOTE_NIST if sp["held_source"] == NEW_SRC
                               else cv.MECHANISM_NOTES.get(sp["held_source"], "")),
        }
        folds.append(cv.finish_fold(
            cache, sp["material"], sp["held_points"], sp["S0"], sp["method"],
            sp["train_err"], sp["trace"], meta,
            alt_locked=cv.S0_KNOWN_V12.get(sp["material"])))

    block = {
        "folds": folds,
        "skipped": loso_skipped,
        "summary": cv.summarise(cv.all_ef_errors(folds), "LOSO (with nist_mds2_3402)"),
        "hold_v12_only_summary": cv.summarise(
            [e for f in folds if f.get("held_source") == "v12_original"
             for e in [p["ef_error_pct"] for p in f["predictions"]]],
            "LOSO (held source = v12_original)"),
        "hold_extended_only_summary": cv.summarise(
            [e for f in folds if f.get("held_source") != "v12_original"
             for e in [p["ef_error_pct"] for p in f["predictions"]]],
            "LOSO (held source = extended literature source)"),
        "in718_confounded_diagnostic": locked_loso.get("in718_confounded_diagnostic"),
        "in718_confounded_diagnostic_note":
            ("carried over verbatim from the locked R13 run: no IN718 point was added or "
             "changed by R13b, so this declared diagnostic is not recomputed."),
    }

    # ---- regression check: folds whose input sets did not change must reproduce R13 ----
    reg = []
    old_by = {(f["material"], f["held_source"]): f for f in locked_loso["folds"]}
    for f in folds:
        key = (f["material"], f["held_source"])
        o = old_by.get(key)
        if o is None:
            reg.append({"fold": "%s/%s" % key, "in_locked_r13": False,
                        "note": "new fold created by the mds2 addition"})
            continue
        same_inputs = (sorted(o.get("train_point_ids", [])) ==
                       sorted(f.get("train_point_ids", [])))
        same_s0 = abs(float(o["S0"]) - float(f["S0"])) < 1e-12
        same_err = (abs(float(o["avg_held_ef_error_pct"]) - float(f["avg_held_ef_error_pct"]))
                    < 1e-12)
        reg.append({
            "fold": "%s/%s" % key, "in_locked_r13": True,
            "train_point_ids_identical": same_inputs,
            "S0_identical": same_s0, "avg_held_ef_error_identical": same_err,
            "R13": {"S0": o["S0"], "avg_held_ef_error_pct": o["avg_held_ef_error_pct"]},
            "R13b": {"S0": f["S0"], "avg_held_ef_error_pct": f["avg_held_ef_error_pct"]},
            "note": ("unchanged inputs: reproduced exactly (this is a real regression check "
                     "on the shared machinery)" if same_inputs and same_s0 and same_err else
                     "inputs changed: the training set now contains the mds2 as-built HT0 "
                     "points, so the fold is expected to move"),
        })
    for r in reg:
        log("regression %-40s inputs_same=%-5s S0_same=%-5s err_same=%-5s  R13=%s -> R13b=%s"
            % (r["fold"], r.get("train_point_ids_identical"), r.get("S0_identical"),
               r.get("avg_held_ef_error_identical"),
               r.get("R13", {}).get("avg_held_ef_error_pct"),
               r.get("R13b", {}).get("avg_held_ef_error_pct")))
    if any(r.get("in_locked_r13") and r.get("train_point_ids_identical")
           and not (r["S0_identical"] and r["avg_held_ef_error_identical"]) for r in reg):
        raise SystemExit("REGRESSION: a fold with identical inputs did not reproduce R13")

    # ---- the mds2 hold-out diagnostic: heat-treated vs as-built held points ----
    nist_fold = next((f for f in folds if f.get("held_source") == NEW_SRC), None)
    diag = None
    if nist_fold:
        rows = []
        for p in nist_fold["predictions"]:
            pt = by_id[p["point_id"]]
            signed = (p["ef_pred"] - p["ef_exp"]) / p["ef_exp"] * 100.0
            rows.append({
                "point_id": p["point_id"],
                "heat_treatment": pt["heat_treatment"],
                "heat_treatment_class": cv.heat_treatment_class(pt),
                "orientation_deg": p["orientation_deg"],
                "ef_exp": p["ef_exp"], "ef_pred": p["ef_pred"],
                "ef_error_pct": p["ef_error_pct"],
                "signed_ef_error_pct": round(signed, 2),
                "uts_exp_MPa": p["uts_exp_MPa"], "uts_pred_MPa": p["uts_pred_MPa"],
                "uts_error_pct": p["uts_error_pct"],
            })

        def sub(pred):
            sel = [r for r in rows if pred(r)]
            if not sel:
                return None
            return {
                "n": len(sel),
                "summary_ef": cv.summarise([r["ef_error_pct"] for r in sel],
                                           "mds2 held subset"),
                "mean_signed_ef_error_pct": round(
                    sum(r["signed_ef_error_pct"] for r in sel) / len(sel), 2),
                "ef_exp_min": min(r["ef_exp"] for r in sel),
                "ef_exp_max": max(r["ef_exp"] for r in sel),
                "uts_exp_min_MPa": min(r["uts_exp_MPa"] for r in sel),
                "uts_exp_max_MPa": max(r["uts_exp_MPa"] for r in sel),
            }

        calib = [p for p in points if p["source_id"] != NEW_SRC and p["material"] == "Ti64"
                 and cv.is_as_built(p)]
        diag = {
            "why": ("the held source mixes 1 as-built and 14 heat-treated conditions while the "
                    "training set is as-built only; splitting the held points by heat-treatment "
                    "class isolates the descriptor gap from the transfer error"),
            "held_S0": nist_fold["S0"],
            "train_sources": nist_fold["train_sources"],
            "as_built_HT0_only": sub(lambda r: r["heat_treatment_class"] == "as-built"),
            "heat_treated_all": sub(lambda r: r["heat_treatment_class"] == "heat-treated"),
            "stress_relief_HT1_only": sub(
                lambda r: r["heat_treatment"].lower().startswith("stress-relief")),
            "hip_conditions_HT2_to_HT14": sub(
                lambda r: r["heat_treatment_class"] == "heat-treated"
                and not r["heat_treatment"].lower().startswith("stress-relief")),
            "as_built_calibration_reference": {
                "n": len(calib),
                "sources": sorted({p["source_id"] for p in calib}),
                "ef_min": min(p["ef"] for p in calib),
                "ef_max": max(p["ef"] for p in calib),
                "uts_min_MPa": min(p["uts_MPa"] for p in calib),
                "uts_max_MPa": max(p["uts_MPa"] for p in calib),
                "note": ("the as-built Ti64 points of the OTHER sources - what S0 was "
                         "calibrated on"),
            },
            "per_point": sorted(rows, key=lambda r: -r["ef_error_pct"]),
            "mechanism_note": MECH_NOTE_NIST,
        }

        # ---- the attribution is MEASURED, not assumed ----
        ab = diag["as_built_HT0_only"]
        ht = diag["heat_treated_all"]
        hip = diag["hip_conditions_HT2_to_HT14"]
        ab_err, ht_err = ab["summary_ef"]["mean_ef_error_pct"], ht["summary_ef"]["mean_ef_error_pct"]
        if ab_err >= 0.8 * ht_err:
            verdict = (
                "CONCLUSION ON ATTRIBUTION: the as-built HT0 points fail almost as badly as the "
                "heat-treated points (%.2f%% vs %.2f%% mean ef error over n=%d and n=%d). Heat "
                "treatment therefore does NOT explain this fold on its own, and attributing the "
                "error to the absent heat-treatment descriptor alone would be wrong. The "
                "dominant cause is the property-range gap, which is present as-built as well: "
                "the S0 calibrated on the v12/sun2024 as-built points does not transfer to this "
                "dataset at ANY heat-treatment state." % (ab_err, ht_err, ab["n"], ht["n"]))
        else:
            verdict = (
                "CONCLUSION ON ATTRIBUTION: the heat-treated points fail markedly worse than the "
                "as-built ones (%.2f%% vs %.2f%% mean ef error over n=%d and n=%d), which "
                "isolates the absent heat-treatment descriptor - rather than a process-window "
                "shift - as the dominant cause." % (ht_err, ab_err, ht["n"], ab["n"]))
        extra = (
            " Measured attribution: as-built HT0 (n=%d) mean ef error %.2f%% (median %.2f%%); "
            "HIP HT2-HT14 (n=%d) mean %.2f%% (median %.2f%%); HIP by cooling class - the 13 "
            "high-ductility treated conditions are the ones the band check flagged. %s"
            % (ab["n"], ab_err, ab["summary_ef"]["median_ef_error_pct"], hip["n"],
               hip["summary_ef"]["mean_ef_error_pct"],
               hip["summary_ef"]["median_ef_error_pct"], verdict))
        diag["mechanism_note"] += extra
        if nist_fold is not None:
            nist_fold["mechanism_note"] += extra

    payload = {
        "model": locked.get("model"),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": ("R13b: re-run the leave-one-source block after adding NIST mds2-3402 as a "
                    "third independent Ti64 source. Nothing in the locked R13 statistics is "
                    "overwritten; the two blocks are reported side by side and the locked file "
                    "is read-only in this script."),
        "scope": {
            "recomputed": ["leave_one_source"],
            "not_recomputed": ["leave_one_material", "leave_one_process",
                               "comparison_with_v12_loocv", "protocol_deviation_check",
                               "orientation_mapping_flag", "failure_mode_analysis"],
            "why": ("mds2-3402 contains only Ti-6Al-4V, so 316L, AlSi10Mg and IN718 folds "
                    "cannot change; their folds are recomputed here anyway as a regression "
                    "check (identical inputs must reproduce the locked numbers), but LOMO, "
                    "LOPO and the paired v12 comparison are left untouched."),
        },
        "dataset": {
            "csv": cv.CSV_PATH,
            "csv_sha256": sha256(cv.CSV_PATH),
            "n_points": len(points),
            "n_sources": len({p["source_id"] for p in points}),
            "n_by_source": {s: sum(1 for p in points if p["source_id"] == s)
                            for s in sorted({p["source_id"] for p in points})},
            "n_by_material": {m: sum(1 for p in points if p["material"] == m)
                              for m in sorted({p["material"] for p in points})},
        },
        "protocol": locked.get("protocol"),
        "cache": {
            "path": cv.CACHE_PATH,
            "n_entries": len(cache),
            "n_entries_locked_r13": locked["prediction_cache_manifest"]["n_entries"],
            "n_entries_added_by_r13b":
                len(cache) - locked["prediction_cache_manifest"]["n_entries"],
            "n_new_runs_this_invocation": n_new,
            "n_failed_runs": n_fail,
            "note": ("shared with the locked R13 run; the wave-1 grid universe is unchanged "
                     "because mds2 introduces no new (material, orientation) pair - it maps "
                     "onto the existing Ti64 0/90 deg columns. All runs added by R13b are "
                     "Ti64 refinements at S0 = 0.5 and 1.28 (3 orientations each); a second "
                     "invocation of this script therefore adds 0 runs."),
        },
        "loso_v13_original": dict(
            locked_loso,
            provenance={
                "source_file": cv.RESULT_PATH,
                "copied_verbatim": True,
                "status": "LOCKED R13 statistics - republished here unchanged, not recomputed",
                "n_points_in_that_dataset": locked["dataset_summary"]["n_points"],
                "n_by_source_in_that_dataset": locked["dataset_summary"]["n_by_source"],
            }),
        "loso_with_nist": block,
        "regression_check_vs_locked": reg,
        "nist_mds2_holdout_diagnostic": diag,
        "run": {"wall_seconds_total": round(time.time() - t0, 1), "workers": args.workers,
                "n_cache_entries": len(cache), "log": LOGPATH,
                "completed": time.strftime("%Y-%m-%d %H:%M:%S")},
    }

    os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    log("")
    log("=" * 78)
    for tag, blob in (("loso_v13_original", locked_loso), ("loso_with_nist", block)):
        s = blob["summary"]
        log("%-20s folds=%d n=%3d mean=%6.2f median=%6.2f pass@10=%d/%d"
            % (tag, len(blob["folds"]), s["n_predictions"], s["mean_ef_error_pct"],
               s["median_ef_error_pct"], s["pass_10pct"], s["n_predictions"]))
    for f in folds:
        alt = f.get("alternative_avg_held_ef_error_pct")
        log("   %-9s held=%-22s n_held=%2d S0=%-10.6g ef_err=%7.2f uts_err=%7s"
            " (alt locked S0 -> %s)"
            % (f["material"], f["held_source"], f["n_held"], f["S0"],
               f["avg_held_ef_error_pct"], f["avg_held_uts_error_pct"], alt))
    if diag:
        for k in ("as_built_HT0_only", "stress_relief_HT1_only",
                  "hip_conditions_HT2_to_HT14", "heat_treated_all"):
            v = diag[k]
            if v:
                log("   mds2 %-26s n=%2d mean ef err=%7.2f%%  median=%7.2f%%"
                    % (k, v["n"], v["summary_ef"]["mean_ef_error_pct"],
                       v["summary_ef"]["median_ef_error_pct"]))
    log("DONE -> %s  (%.1fs)" % (OUTPATH, time.time() - t0))
    log.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
