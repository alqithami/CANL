"""Command line interface: ``caenl <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_validate(args: argparse.Namespace) -> int:
    from .plan import Plan
    from .validate import validate_plan

    plan = Plan.load(args.plan)
    report = validate_plan(plan, check_data=args.check_data, check_env=args.check_env)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["valid"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_campaign

    return run_campaign(
        args.plan,
        resume=not args.no_resume,
        only_stage=args.only_stage,
        from_stage=args.from_stage,
        parallel_override=args.parallel,
        gpus=args.gpus,
        keep_going=not args.fail_fast,
        allow_drift=args.allow_drift,
        dry_run=args.dry_run,
        report=not args.no_report,
        max_retries=args.max_retries,
    )


def cmd_run_job(args: argparse.Namespace) -> int:
    from .tasks import resolve
    from .utils.jobctx import JobContext
    from .utils.seeding import seed_everything

    ctx = JobContext(args.job_dir)
    seed_everything(ctx.seed, deterministic=bool(ctx.get("reproducibility.deterministic_algorithms", False)), cudnn_benchmark=bool(ctx.get("reproducibility.cudnn_benchmark", True)))
    ctx.start()
    try:
        fn = resolve(ctx.config["type"])
        summary = fn(ctx)
        ctx.finish(summary or {})
        return 0
    except BaseException as exc:  # noqa: BLE001
        ctx.fail(exc)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    from .runner import status_report

    print(status_report(args.root, details=args.details))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report.aggregate import run_report

    try:
        out = run_report(Path(args.root), strict=args.strict, extra_roots=[Path(r) for r in (args.extra_roots or [])])
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from .tasks.prepare_data import prepare

    for ds in args.dataset:
        info = prepare(ds, data_root=Path(args.data_root), cache_root=Path(args.cache_root), options=json.loads(args.options) if args.options else {})
        print(json.dumps({ds: info}, indent=2, default=str))
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from .plan import Plan
    from .validate import preflight

    plan = Plan.load(args.plan)
    rep = preflight(plan, download=args.download)
    print(json.dumps(rep, indent=2, default=str))
    for c in rep["checks"]:
        mark = "OK  " if c["ok"] else ("FAIL" if c["required"] else "warn")
        print(f"[preflight] {mark} {c['name']}: {c['detail']}", file=sys.stderr)
    print(f"[preflight] {'READY' if rep['ok'] else 'NOT READY: ' + ', '.join(rep['failed_required'])}", file=sys.stderr)
    return 0 if rep.get("ok", False) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="caenl", description="CAENL Franklin Open revision pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a plan and its protocol")
    v.add_argument("--plan", required=True)
    v.add_argument("--check-data", action="store_true")
    v.add_argument("--check-env", action="store_true")
    v.set_defaults(fn=cmd_validate)

    r = sub.add_parser("run", help="run (or resume) a campaign")
    r.add_argument("--plan", required=True)
    r.add_argument("--no-resume", action="store_true", help="re-queue jobs that already succeeded (default: skip them). Each job still restarts from its own checkpoints; delete a job directory (or the stage's jobs/ directory) to recompute it from scratch")
    r.add_argument("--only-stage")
    r.add_argument("--from-stage")
    r.add_argument("--parallel", type=int, default=None, help="override per-stage parallelism")
    r.add_argument("--gpus", default=None, help="comma separated GPU ids, e.g. 0 or 0,1,2,3")
    r.add_argument("--fail-fast", action="store_true", help="stop launching new jobs after the first failed stage (default: only the stages that depend on a failed stage are blocked; independent stages continue)")
    r.add_argument("--allow-drift", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--max-retries", type=int, default=1, help="automatic re-queue of a failed job (transient CUDA/OOM errors)")
    r.set_defaults(fn=cmd_run)

    j = sub.add_parser("run-job", help="(internal) execute one job directory")
    j.add_argument("--job-dir", required=True)
    j.set_defaults(fn=cmd_run_job)

    s = sub.add_parser("status", help="print campaign status")
    s.add_argument("--root", required=True)
    s.add_argument("--details", action="store_true")
    s.set_defaults(fn=cmd_status)

    rp = sub.add_parser("report", help="aggregate results and regenerate tables/figures")
    rp.add_argument("--root", required=True)
    rp.add_argument("--strict", action="store_true", help="publication-readiness check: every expected job succeeded with finite required metrics under the frozen protocol, hashes stable, report artefacts present (non-zero exit otherwise)")
    rp.add_argument("--extra-roots", nargs="*", default=None, help="campaign roots of other pods (same protocol) to merge into this report, e.g. the ViT pod")
    rp.set_defaults(fn=cmd_report)

    d = sub.add_parser("prepare-data", help="download/caches datasets")
    d.add_argument("--dataset", nargs="+", required=True)
    d.add_argument("--data-root", default="data")
    d.add_argument("--cache-root", default="cache")
    d.add_argument("--options", default=None, help="JSON dict of dataset options")
    d.set_defaults(fn=cmd_prepare_data)

    pf = sub.add_parser("preflight", help="check environment, data and disk before a campaign")
    pf.add_argument("--plan", required=True)
    pf.add_argument("--download", action="store_true", help="download/caches missing datasets")
    pf.set_defaults(fn=cmd_preflight)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
