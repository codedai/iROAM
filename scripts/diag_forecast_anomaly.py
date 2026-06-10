"""Diagnose the reported gap-vs-probability inversion in the forecast service.

Pulls a route-29 slice at a chosen t_ref, runs the live predictor over every
eligible bus, and prints:
  * each bus's current forward_gap, gap_closure_m_s, stop_idx, max_prob,
    max_prob_step, and the FULL 30-horizon probability vector;
  * Spearman correlation between forward_gap and max_prob across the slice
    (we expect *negative* correlation: bigger gap → less likely to bunch);
  * the bus(es) with the highest max_prob, for closer inspection.

Run:
    python -m scripts.diag_forecast_anomaly --date 2026-05-30 --dir 0 --t-ref 720
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np

from apps.analytics.stop_projection import compute_route_stops
from apps.api.routers.iroam import _group_into_buses
from apps.api.services.bunching_predictor import get_predictor, reset_cache
from apps.api.services.forecast import run_forecast
from db.queries.iroam import fetch_trajectories_for_slice
from db.session import SessionLocal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", default="29")
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--dir", type=int, default=0)
    p.add_argument("--t-ref", type=float, required=True, help="minute of day in local TZ")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    reset_cache()
    pred = get_predictor()
    print(f"Predictor: {pred.metadata.get('model_type')}  "
          f"feature_set={pred.metadata.get('feature_set')}  "
          f"calibrated={pred.metadata.get('calibrated')}  "
          f"seq_len={pred.seq_len}  pred_len={pred.pred_len}  n_channels={pred.n_channels}",
          flush=True)

    rs = compute_route_stops(args.route, args.dir)
    if rs is None:
        raise SystemExit(f"no shape for route={args.route} dir={args.dir}")
    with SessionLocal() as s:
        rows = fetch_trajectories_for_slice(
            s, service_date=args.date, route_id=args.route, direction_id=args.dir,
        )
    buses = _group_into_buses(rows, rs)
    res = run_forecast(
        buses, num_stops=len(rs.stops), t_ref_min=args.t_ref,
        route_shape_length_m=float(rs.shape_length_m),
    )
    print(f"running={res.num_running}  eligible={res.num_eligible}  total={res.num_buses_total}",
          flush=True)

    eligible = [r for r in res.per_bus if r["eligible"]]
    if not eligible:
        print("(no eligible buses at this t_ref)")
        return

    gaps = np.array([r["forward_gap_m"] for r in eligible], dtype=float)
    closures = np.array([r["gap_closure_m_s"] for r in eligible], dtype=float)
    max_probs = np.array([r["max_prob"] for r in eligible], dtype=float)
    stop_idx = np.array([r["stop_idx"] for r in eligible], dtype=float)

    # Spearman correlation between gap and max_prob.
    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
        ar = ar.astype(float); br = br.astype(float)
        if ar.std() == 0 or br.std() == 0:
            return float("nan")
        return float(np.corrcoef(ar, br)[0, 1])

    print(f"Spearman(forward_gap_m, max_prob)    = {_spearman(gaps, max_probs):+.3f}   "
          f"(expected: NEGATIVE)", flush=True)
    print(f"Spearman(gap_closure_m_s, max_prob)  = {_spearman(closures, max_probs):+.3f}   "
          f"(expected: POSITIVE)", flush=True)
    print(f"Spearman(stop_idx, max_prob)         = {_spearman(stop_idx, max_probs):+.3f}",
          flush=True)
    print()

    # Sort by max_prob desc, print top-K with full per-horizon.
    eligible_sorted = sorted(eligible, key=lambda r: -(r["max_prob"] or 0))
    print(f"Top {args.top} by max_prob:")
    print(f"  {'veh_id':>8s}  {'p_max':>5s} {'@h':>3s}  {'gap_m':>7s}  {'Δm/s':>6s}  {'stop':>5s}")
    for r in eligible_sorted[: args.top]:
        print(f"  {str(r['vehicle_id']):>8s}  {r['max_prob']:.2f} {r['max_prob_step']+1:>3d}  "
              f"{r['forward_gap_m']:>7.0f}  {r['gap_closure_m_s']:>+6.2f}  {r['stop_idx']:>5.1f}")
    print()

    # Full per-horizon for the single highest-prob bus.
    top = eligible_sorted[0]
    print(f"Per-horizon prob for veh {top['vehicle_id']}:")
    print("  " + ", ".join(f"+{h+1}:{p:.2f}" for h, p in enumerate(top["per_horizon"])))


if __name__ == "__main__":
    main()
