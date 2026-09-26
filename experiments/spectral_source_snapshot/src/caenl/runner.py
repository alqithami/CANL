"""Sequential/parallel, restartable campaign runner.

* Each job runs in its own subprocess (``caenl run-job --job-dir ...``) for isolation.
* Resume is the default: jobs whose ``status.json`` says ``succeeded`` and whose ``summary.json``
  exists are skipped; failed or interrupted jobs are restarted (tasks resume from their own
  checkpoints).  ``--no-resume`` re-queues the succeeded jobs too (each still restarts from its own
  checkpoints; deleting a job directory is the way to recompute it from scratch).
* After every stage the publication gate (:mod:`caenl.publication`) validates the stage against
  the plan's expected job matrix.  A stage that fails or does not validate is marked ``failed``,
  the stages depending on it are ``blocked`` and independent stages continue (``--fail-fast``
  stops every launch instead); a report refresh that fails stops every launch.
* One runner per campaign root (``<root>/.runner.lock``); jobs left ``running`` by a dead runner
  are detected through their pid and restarted.
* Jobs never run with a configuration that differs from the one frozen in ``job.json``;
  a changed plan for an already-started job is refused (``--allow-drift`` overrides and
  records the drift).
* Scheduling is global and dependency-aware: jobs of every stage whose ``depends_on`` stages have
  been validated share the GPU pool (``--gpus``); ``parallel`` caps a stage's concurrency, ``gpu_share``
  lets small jobs share a GPU, ``exclusive`` stages run alone.  Plans without ``depends_on`` keep the
  sequential stage order.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from .plan import Plan
from .utils.io import atomic_write_json, dump_yaml, read_json, sha256_file, stable_hash, utc_now
from .utils.telemetry import environment_snapshot


def job_dir_for(plan: Plan, stage_name: str, job_id: str) -> Path:
    return plan.campaign_root / "stages" / stage_name / "jobs" / job_id.replace("/", "__")


def job_state(job_dir: Path) -> str:
    st = job_dir / "status.json"
    if not st.exists():
        return "pending"
    try:
        state = read_json(st).get("state", "pending")
    except Exception:
        return "pending"
    if state == "succeeded" and not (job_dir / "summary.json").exists():
        return "failed"
    return state


def _source_snapshot(project_root: Path) -> dict[str, Any]:
    """Hash the executable source/config snapshot used by a campaign.

    Source archives are not necessarily Git checkouts on the IBM VM, so a deterministic
    allow-listed tree hash is recorded independently of Git metadata.
    """
    project_root = Path(project_root).resolve()
    allowed_dirs = ("src", "configs", "scripts", "requirements", "tests", "docker", "docs")
    allowed_files = ("pyproject.toml", "requirements.txt", "README.md")
    ignored_parts = {"__pycache__", ".pytest_cache", ".venv", ".git"}
    records: list[dict[str, Any]] = []
    candidates: list[Path] = []
    for name in allowed_files:
        p = project_root / name
        if p.is_file():
            candidates.append(p)
    for name in allowed_dirs:
        d = project_root / name
        if d.is_dir():
            candidates.extend(p for p in d.rglob("*") if p.is_file())
    for p in sorted(set(candidates), key=lambda x: x.relative_to(project_root).as_posix()):
        rel = p.relative_to(project_root)
        if any(part in ignored_parts for part in rel.parts) or p.suffix in {".pyc", ".pyo"}:
            continue
        records.append({"path": rel.as_posix(), "size": p.stat().st_size, "sha256": sha256_file(p)})
    digest = stable_hash(records, 64)
    return {"project_root": str(project_root), "tree_sha256": digest, "file_count": len(records), "files": records}


def freeze_plan(plan: Plan) -> Path:
    root = plan.campaign_root
    root.mkdir(parents=True, exist_ok=True)
    plan_copy = root / "PLAN.yaml"
    fp = plan.fingerprint()
    fingerprint_path = root / "PLAN.fingerprint"
    if plan_copy.exists():
        old = ""
        if fingerprint_path.exists():
            old = fingerprint_path.read_text().strip().split()[0]
        elif (root / "PLAN.sha256").exists():
            legacy = (root / "PLAN.sha256").read_text().strip().split()[0]
            if len(legacy) == 16:  # v5.4.1 stored the short fingerprint under a misleading name
                old = legacy
        if old and old != fp:
            (root / "PLAN_DRIFT.log").open("a").write(f"{utc_now()} plan fingerprint changed {old} -> {fp}\n")
    dump_yaml({"plan": plan.raw, "protocol": plan.protocol}, plan_copy)
    fingerprint_path.write_text(f"{fp}  PLAN.yaml\n")
    (root / "PLAN.sha256").write_text(f"{sha256_file(plan_copy)}  PLAN.yaml\n")
    if plan.protocol_path is not None:
        frozen_protocol = root / "frozen_protocol.yaml"
        dump_yaml(plan.protocol, frozen_protocol)
        (root / "frozen_protocol.sha256").write_text(f"{sha256_file(frozen_protocol)}  frozen_protocol.yaml\n")
    project_root = Path(os.environ.get("CAENL_PROJECT_DIR", str(Path(__file__).resolve().parents[2]))).resolve()
    source_snapshot = _source_snapshot(project_root)
    source_snapshot_path = root / "SOURCE_SNAPSHOT.json"
    atomic_write_json(source_snapshot_path, source_snapshot)
    source_snapshot_file_sha256 = sha256_file(source_snapshot_path)
    (root / "SOURCE_SNAPSHOT.sha256").write_text(f"{source_snapshot_file_sha256}  SOURCE_SNAPSHOT.json\n")
    manifest = root / "campaign_manifest.json"
    if not manifest.exists():
        atomic_write_json(
            manifest,
            {
                "campaign_id": plan.campaign_id,
                "created_at": utc_now(),
                "plan_fingerprint": fp,
                "source_tree_sha256": source_snapshot["tree_sha256"],
                "source_snapshot_file_sha256": source_snapshot_file_sha256,
                "source_snapshot_file_count": source_snapshot["file_count"],
                "environment": environment_snapshot(),
            },
        )
    return root


def materialize_jobs(plan: Plan, stage: dict[str, Any], jobs: list[dict[str, Any]], allow_drift: bool = False) -> list[Path]:
    dirs: list[Path] = []
    for job in jobs:
        jd = job_dir_for(plan, stage["name"], job["job_id"])
        jd.mkdir(parents=True, exist_ok=True)
        jf = jd / "job.json"
        job = dict(job)
        job["job_dir"] = str(jd)
        new_hash = stable_hash({k: v for k, v in job.items() if k != "job_dir"}, 16)
        if jf.exists():
            old = read_json(jf)
            old_hash = stable_hash({k: v for k, v in old.items() if k != "job_dir"}, 16)
            if old_hash != new_hash:
                state = job_state(jd)
                if state in ("running", "succeeded") and not allow_drift:
                    raise RuntimeError(
                        f"job {job['job_id']} already {state} with a different configuration "
                        f"({old_hash} != {new_hash}). Use a new campaign_id or --allow-drift."
                    )
                (jd / "CONFIG_DRIFT.log").open("a").write(f"{utc_now()} {old_hash} -> {new_hash}\n")
                atomic_write_json(jf, job)
        else:
            atomic_write_json(jf, job)
        dirs.append(jd)
    return dirs


def thread_env(parallel: int) -> dict[str, str]:
    """CPU thread caps for one job when ``parallel`` jobs share the pod (never exceed the core count)."""
    cores = os.cpu_count() or 4
    per_job = max(1, cores // max(1, parallel))
    out = {}
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        out[var] = os.environ.get(var, str(per_job))
    out.setdefault("TOKENIZERS_PARALLELISM", os.environ.get("TOKENIZERS_PARALLELISM", "false"))
    return out


def _launch(job_dir: Path, gpu: Optional[str], extra_env: dict[str, str], parallel: int = 1) -> subprocess.Popen:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.update(thread_env(parallel))
    env["CAENL_PARALLEL"] = str(max(1, int(parallel)))  # host-RAM budget per job = fraction of RAM / parallel
    env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log = (job_dir / "log.txt").open("a", encoding="utf-8")
    log.write(f"\n===== launch {utc_now()} gpu={gpu} =====\n")
    log.flush()
    cmd = [sys.executable, "-m", "caenl.cli", "run-job", "--job-dir", str(job_dir)]
    try:
        # Avoid ``fork`` in the long-lived parent process whenever the platform
        # can use ``posix_spawn``.  PyTorch/OpenMP may own background threads;
        # repeatedly forking such a process can deadlock before ``exec``.  The
        # runner is already started from the repository root, so inheriting cwd
        # is equivalent to the former explicit ``cwd=Path.cwd()``.
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            close_fds=False,
        )
    finally:
        # Popen duplicates the descriptor for the child.  Closing the parent-side
        # handle prevents descriptor accumulation during long multi-job campaigns
        # and guarantees that log files are not kept open by the runner itself.
        log.close()
    return proc


def write_status_md(plan: Plan, stage_rows: list[dict[str, Any]]) -> None:
    lines = [f"# Campaign {plan.campaign_id}", "", f"Updated: {utc_now()}", "", "| stage | total | succeeded | running | failed | pending |", "|---|---|---|---|---|---|"]
    for r in stage_rows:
        lines.append(f"| {r['stage']} | {r['total']} | {r['succeeded']} | {r['running']} | {r['failed']} | {r['pending']} |")
    (plan.campaign_root / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_counts(dirs: list[Path]) -> dict[str, int]:
    c = {"succeeded": 0, "running": 0, "failed": 0, "pending": 0}
    for d in dirs:
        c[job_state(d) if job_state(d) in c else "pending"] += 1
    return c


def verify_resumed(stage: dict[str, Any], dirs: list[Path]) -> list[str]:
    """After an interruption stage: every job must have resumed from a checkpoint (and, for Full MACC
    vision jobs, restored its pending acquisition action so that every round was rewarded)."""
    errors: list[str] = []
    want = dict(stage.get("interrupt", {}).get("verify", {}) or {})
    for d in dirs:
        if not (d / "INTERRUPTED.json").exists():
            errors.append(f"{d.name}: no interruption marker (the interruption pass did not run)")
            continue
        events = []
        ev = d / "events.jsonl"
        if ev.exists():
            for line in ev.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        resumed = [e for e in events if e.get("kind") == "resumed"]
        if not resumed:
            errors.append(f"{d.name}: no 'resumed' event after the forced interruption (checkpoint restore did not happen)")
            continue
        if "from_round_min" in want:
            if max(int(e.get("from_round", -1)) for e in resumed) < int(want["from_round_min"]):
                errors.append(f"{d.name}: resumed from round {[e.get('from_round') for e in resumed]}, expected >= {want['from_round_min']}")
        if want.get("full_macc_acquisition_updates"):
            try:
                cfg = read_json(d / "job.json")
                if cfg.get("method", {}).get("regularizer") == "full_macc":
                    traj = read_json(d / "artifacts" / "controller_trajectory.json")
                    acq = [h for h in traj.get("history", []) if h.get("kind") == "acquisition_update"]
                    rounds = int(cfg.get("active_learning", {}).get("rounds", 0))
                    if len(acq) != rounds or any(int(h.get("queries", 0)) <= 0 for h in acq):
                        errors.append(f"{d.name}: Full MACC rewarded {len(acq)} acquisition actions (expected {rounds}); pending action not restored across the interruption")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{d.name}: cannot verify Full MACC pending restoration: {exc!r}")
    return errors


# ----------------------------------------------------------------------------- scheduler
class _Running:
    __slots__ = ("proc", "dir", "gpu", "stage", "phase", "share", "cpu_only")

    def __init__(self, proc: subprocess.Popen, d: Path, gpu: Optional[str], stage: str, phase: str, share: float, cpu_only: bool = False):
        self.proc, self.dir, self.gpu, self.stage, self.phase, self.share, self.cpu_only = proc, d, gpu, stage, phase, share, cpu_only


class _StageRun:
    """Bookkeeping for one stage inside the global scheduler."""

    def __init__(self, stage: dict[str, Any], jobs: list[dict[str, Any]], dirs: list[Path], deps: list[str], parallel: int, share: int, exclusive: bool, cpu_only: bool = False):
        self.stage, self.jobs, self.dirs, self.deps = stage, jobs, dirs, deps
        self.name = stage["name"]
        self.type = stage["type"]
        self.parallel, self.share, self.exclusive, self.cpu_only = parallel, share, exclusive, cpu_only
        self.state = "waiting"  # waiting -> ready -> running -> validated | failed | skipped
        self.pending: deque[tuple[Path, str]] = deque()  # (dir, phase) with phase in {"interrupt", "normal"}
        self.attempts: dict[Path, int] = {}
        self.running = 0
        self.failures = 0
        self.interrupt_env = {str(k): str(v) for k, v in (stage.get("interrupt", {}).get("env", {}) or {}).items()}

    def queue_jobs(self, resume: bool) -> None:
        for d in self.dirs:
            st = job_state(d)
            if resume and st == "succeeded":
                continue
            if st == "running":
                if _stale_running(d):
                    atomic_write_json(d / "status.json", {"state": "failed", "job_id": d.name, "time": utc_now(), "error": "stale 'running' status from a dead runner; restarted"})
                else:
                    raise RuntimeError(f"{d} is still running under another process (pid in status.json); stop it before rerunning")
            if self.interrupt_env and not (d / "INTERRUPTED.json").exists():
                self.pending.append((d, "interrupt"))
            else:
                self.pending.append((d, "normal"))

    def done(self) -> bool:
        return not self.pending and self.running == 0


def resolve_dependencies(stages: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Explicit ``depends_on`` lists, or the previous stage when a stage declares none (sequential plans)."""
    deps: dict[str, list[str]] = {}
    names = [s["name"] for s in stages]
    for i, st in enumerate(stages):
        if "depends_on" in st:
            raw = st.get("depends_on") or []
            raw = [raw] if isinstance(raw, str) else list(raw)
            for d in raw:
                if d not in names:
                    raise ValueError(f"stage {st['name']} depends on unknown stage {d!r}")
            deps[st["name"]] = raw
        elif st["type"] == "aggregate":
            deps[st["name"]] = [n for n in names if n != st["name"]]  # the report needs everything
        else:
            deps[st["name"]] = names[i - 1 : i]
    return deps


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError, OverflowError):
        return False


def _stale_running(job_dir: Path) -> bool:
    """A status.json left at ``running`` by a runner that died (no live process behind it)."""
    if job_state(job_dir) != "running":
        return False
    try:
        pid = read_json(job_dir / "status.json").get("pid")
    except Exception:
        pid = None
    return not _pid_alive(pid)


def run_campaign(
    plan_path: str,
    resume: bool = True,
    only_stage: Optional[str] = None,
    from_stage: Optional[str] = None,
    parallel_override: Optional[int] = None,
    gpus: Optional[str] = None,
    keep_going: bool = True,
    allow_drift: bool = False,
    dry_run: bool = False,
    report: bool = True,
    max_retries: int = 1,
    poll_s: float = 5.0,
) -> int:
    """Global, dependency-aware scheduler.

    * Every stage declares ``depends_on`` (default: the previous stage, i.e. the sequential semantics of
      the smoke plans); jobs of all dependency-satisfied stages share the GPU pool at once, so the tail
      of a long stage never leaves GPUs idle.
    * ``parallel`` caps a stage's concurrent jobs; ``gpu_share`` (default 1) is how many jobs of the
      stage may share one GPU (small CIFAR jobs); ``exclusive: true`` stages (and timing benchmarks)
      run with nothing else on the GPUs and have the lowest priority; ``cpu_only`` stages take no GPU.
    * A finished stage is validated by the publication gate (and the resume check for interruption
      stages) before its dependants start.  A failed job (after ``max_retries``), a failed gate or a
      failed report refresh marks the stage *failed*: its dependants (transitively, including the
      final ``aggregate``) are blocked, independent stages keep running, and the campaign exits
      non-zero.  ``--fail-fast`` stops every further launch at the first failure instead.
    """
    import fcntl
    import signal

    plan = Plan.load(plan_path)
    root = freeze_plan(plan)
    print(f"[runner] campaign root: {root}", flush=True)
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()] if gpus else None
    if gpu_list is None and not dry_run:
        try:
            import torch

            if torch.cuda.device_count() > 1:
                print(f"[runner] ERROR: {torch.cuda.device_count()} GPUs are visible but --gpus was not given; every job would land on GPU 0. Pass --gpus 0,1,...", flush=True)
                return 2
        except Exception:
            pass
    expanded = plan.expand()
    deps = resolve_dependencies([st for st, _ in expanded])
    gate = bool(plan.raw.get("publication_gate", True))
    stages: dict[str, _StageRun] = {}
    status_rows: list[dict[str, Any]] = []
    order: list[str] = []
    for stage, jobs in expanded:
        dirs = materialize_jobs(plan, stage, jobs, allow_drift=allow_drift)
        parallel = max(1, int(parallel_override or stage.get("parallel", 1)))
        exclusive = stage["type"] in ("aggregate", "vision_overhead") or bool(stage.get("exclusive", False))
        cpu_only = bool(stage.get("cpu_only", False)) and not exclusive
        share = 1 if exclusive else max(1, int(stage.get("gpu_share", 1)))
        if exclusive:
            parallel = 1
        if gpu_list is None:
            share = max(1, parallel)  # one virtual device with `parallel` slots (CPU / unconstrained runs)
        sr = _StageRun(stage, jobs, dirs, deps[stage["name"]], parallel, share, exclusive, cpu_only)
        stages[stage["name"]] = sr
        order.append(stage["name"])
        row = {"stage": stage["name"], "total": len(dirs)}
        row.update(stage_counts(dirs))
        status_rows.append(row)
    write_status_md(plan, status_rows)
    if dry_run:
        for name in order:
            sr = stages[name]
            print(f"stage {name} ({sr.type}): {len(sr.jobs)} jobs, parallel={sr.parallel}, gpu_share={sr.share}, cpu_only={sr.cpu_only}, exclusive={sr.exclusive}, depends_on={sr.deps}")
            for j in sr.jobs:
                print(f"   - {j['job_id']}  [{job_state(job_dir_for(plan, name, j['job_id']))}]")
        return 0
    # --- one runner per campaign root
    lock_path = root / ".runner.lock"
    lock_fh = open(lock_path, "a+")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[runner] ERROR: another runner holds {lock_path}; stop it first (two runners would launch the same jobs twice)", flush=True)
        return 2
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(f"pid={os.getpid()} started={utc_now()}\n")
    lock_fh.flush()
    # --- stage selection (--only-stage / --from-stage): unselected stages count as satisfied dependencies
    selected = set(order)
    if only_stage:
        if only_stage not in stages:
            raise ValueError(f"unknown stage {only_stage}")
        selected = {only_stage}
    elif from_stage:
        if from_stage not in stages:
            raise ValueError(f"unknown stage {from_stage}")
        selected = set(order[order.index(from_stage):])
    for name in order:
        if name not in selected:
            stages[name].state = "skipped"
    devices: list[Optional[str]] = list(gpu_list) if gpu_list else [None]
    load: dict[Optional[str], float] = {g: 0.0 for g in devices}
    running: list[_Running] = []
    stop_launching = False
    overall_ok = True
    report_thread: Optional[threading.Thread] = None
    report_requested = False
    report_failed: list[str] = []

    def terminate(signum, frame):  # SIGTERM/SIGINT: stop the children, they resume from their checkpoints later
        print(f"[runner] signal {signum}: terminating {len(running)} running job(s) and exiting", flush=True)
        for r in running:
            try:
                r.proc.terminate()
            except Exception:
                pass
        for r in running:
            try:
                r.proc.wait(timeout=30)
            except Exception:
                try:
                    r.proc.kill()
                except Exception:
                    pass
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def satisfied(sr: _StageRun) -> bool:
        return all(stages[d].state in ("validated", "skipped") for d in sr.deps)

    def blocked(sr: _StageRun) -> bool:
        return any(stages[d].state in ("failed", "blocked") for d in sr.deps)

    def refresh_rows() -> None:
        for row in status_rows:
            row.update(stage_counts(stages[row["stage"]].dirs))
        write_status_md(plan, status_rows)

    def fail_stage(sr: _StageRun, why: str) -> None:
        nonlocal overall_ok, stop_launching
        sr.state = "failed"
        overall_ok = False
        print(f"[runner] stage {sr.name} FAILED: {why}", flush=True)
        if not keep_going:
            stop_launching = True
            print("[runner] --fail-fast: no further jobs are launched", flush=True)
        else:
            print(f"[runner] stages depending on {sr.name} are blocked; independent stages continue. Fix the cause and rerun the same command (resume is the default).", flush=True)

    def report_in_flight() -> bool:
        return report_thread is not None and report_thread.is_alive()

    def refresh_report() -> None:
        """Rebuild the campaign report from the validated jobs (in a thread, so running jobs keep being polled).

        While a refresh is in flight no new job is launched: a report that cannot be built from validated
        jobs is a pipeline bug and must stop the campaign *before* further GPU hours are spent.  A refresh
        requested while another one runs is deferred, not dropped."""
        nonlocal report_thread, report_requested
        if not (report and plan.report_after_each_stage):
            return
        if report_in_flight():
            report_requested = True
            return
        report_requested = False

        def _work() -> None:
            try:
                from .report.aggregate import run_report

                run_report(root, quick=True)  # monitoring refresh: capped replicates, see run_report
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                report_failed.append(repr(exc))

        report_thread = threading.Thread(target=_work, daemon=True, name="caenl-report")
        report_thread.start()

    def finish_stage(sr: _StageRun) -> None:
        if sr.failures:
            fail_stage(sr, f"{sr.failures} job(s) failed after {max_retries} retr{'y' if max_retries == 1 else 'ies'}")
            return
        if sr.interrupt_env:
            errs = verify_resumed(sr.stage, sr.dirs)
            atomic_write_json(root / "stages" / sr.name / "RESUME_CHECK.json", {"ok": not errs, "errors": errs, "time": utc_now()})
            if errs:
                for e in errs:
                    print(f"[runner] RESUME CHECK FAILED: {e}", flush=True)
                fail_stage(sr, "forced-interruption resume check failed")
                return
            print(f"[runner] stage {sr.name}: every job resumed from its checkpoint after the forced interruption", flush=True)
        if (gate or bool(sr.stage.get("expect_failure", False))) and sr.type != "aggregate":
            from .publication import validate_stage

            rep = validate_stage(plan, sr.stage, sr.jobs, write_manifest=True)
            if bool(sr.stage.get("expect_failure", False)):
                if rep["ok"]:
                    fail_stage(sr, "expected to fail publication validation but passed: the gate is not working")
                    return
                print(f"[runner] stage {sr.name} rejected by the publication gate as expected ({rep['errors'][:2]})", flush=True)
                sr.state = "validated"
                return
            if not rep["ok"]:
                for e in rep["errors"][:20]:
                    print(f"[runner]    - {e}", flush=True)
                fail_stage(sr, f"publication validation ({rep['n_ok']}/{rep['n_expected']} jobs valid; see {root / 'stages' / sr.name / 'STAGE_MANIFEST.json'})")
                return
            print(f"[runner] stage {sr.name} validated: {rep['n_ok']}/{rep['n_expected']} jobs", flush=True)
        sr.state = "validated"
        if sr.type != "aggregate":
            refresh_report()

    def try_launch() -> bool:
        """Launch at most one job; returns True if something was started."""
        if stop_launching or report_in_flight():
            return False  # launches wait for a running report refresh (its failure must stop the campaign first)
        gpu_jobs = [r for r in running if not r.cpu_only]
        if any(stages[r.stage].exclusive for r in gpu_jobs):
            return False  # an exclusive (timing / aggregate) job owns the GPUs
        # exclusive stages have the lowest priority: they run only when the GPUs are empty and no other ready
        # stage has work left (natural gaps, e.g. while an init stage validates, or the end of the campaign)
        candidates = [stages[n] for n in order if stages[n].state == "running" and stages[n].pending and stages[n].running < stages[n].parallel]
        non_exclusive = [sr for sr in candidates if not sr.exclusive]
        full_gpu_waiting = False  # a higher-priority whole-GPU stage is waiting for capacity
        for sr in non_exclusive + [sr for sr in candidates if sr.exclusive]:
            if sr.exclusive and (gpu_jobs or any(not c.cpu_only for c in non_exclusive)):
                continue
            if sr.cpu_only:
                need, gpu = 0.0, None
            else:
                need = 1.0 / sr.share
                free = [g for g in devices if load[g] + need <= 1.0 + 1e-9]
                if sr.share > 1 and full_gpu_waiting:
                    free = [g for g in free if load[g] > 0]  # fill partially used cards only; keep empty ones for whole-GPU jobs
                if not free:
                    if sr.share == 1:
                        full_gpu_waiting = True
                    continue
                gpu = max(free, key=lambda g: load[g])  # best fit: pack shared jobs, leave whole GPUs free
            d, phase = sr.pending.popleft()
            extra = {"CAENL_STAGE": sr.name}
            if sr.cpu_only:
                extra["CUDA_VISIBLE_DEVICES"] = ""  # data preparation never touches a GPU
            if phase == "interrupt":
                extra.update(sr.interrupt_env)
            n_cap = min(sr.parallel, len(devices) * sr.share) if gpu_list else sr.parallel
            proc = _launch(d, gpu, extra, parallel=max(1, n_cap))
            running.append(_Running(proc, d, gpu, sr.name, phase, need, cpu_only=sr.cpu_only))
            if not sr.cpu_only:
                load[gpu] += need
            sr.running += 1
            print(f"[runner] started {sr.name}/{d.name} (gpu={gpu}{', interruption pass' if phase == 'interrupt' else ''}) pid={proc.pid}", flush=True)
            return True
        return False

    try:
        while True:
            # promote dependency-satisfied stages; block stages whose dependencies failed
            for name in order:
                sr = stages[name]
                if sr.state == "waiting" and blocked(sr):
                    sr.state = "blocked"
                    overall_ok = False
                    print(f"[runner] stage {name} blocked (a dependency failed)", flush=True)
                elif sr.state == "waiting" and satisfied(sr):
                    sr.state = "running"
                    sr.queue_jobs(resume)
                    print(f"[runner] === stage {name} ({sr.type}): {len(sr.jobs)} jobs, parallel={sr.parallel}, gpu_share={sr.share} ===", flush=True)
                    if sr.done():
                        finish_stage(sr)
            if report_failed and not stop_launching:
                # a report that cannot be built from validated jobs is a pipeline bug: stop, do not keep burning GPU hours
                print(f"[runner] report refresh FAILED: {report_failed[-1]}; no further jobs are launched", flush=True)
                overall_ok = False
                stop_launching = True
            if report_requested and not report_in_flight() and not stop_launching:
                refresh_report()  # a refresh deferred while the previous one was running
            while try_launch():
                pass
            if not running and not report_in_flight():
                if all(stages[n].state in ("validated", "failed", "skipped", "blocked") for n in order) or stop_launching:
                    break
                if not any(stages[n].state == "running" for n in order):
                    waiting = [n for n in order if stages[n].state == "waiting"]
                    print(f"[runner] deadlock: stages {waiting} wait for dependencies that can no longer complete", flush=True)
                    overall_ok = False
                    break
                if not any(stages[n].pending for n in order if stages[n].state == "running"):
                    print("[runner] nothing runnable although stages are pending (misconfigured parallel/gpu_share?)", flush=True)
                    overall_ok = False
                    break
            time.sleep(poll_s)
            still: list[_Running] = []
            for r in running:
                rc = r.proc.poll()
                if rc is None:
                    still.append(r)
                    continue
                if not r.cpu_only:
                    load[r.gpu] = max(0.0, load[r.gpu] - r.share)
                sr = stages[r.stage]
                sr.running -= 1
                st = job_state(r.dir)
                if r.phase == "interrupt":
                    atomic_write_json(r.dir / "INTERRUPTED.json", {"time": utc_now(), "env": sr.interrupt_env, "rc": rc, "state_after": st})
                    if st == "succeeded":
                        print(f"[runner] WARNING {sr.name}/{r.dir.name}: the interruption hook did not trigger (job finished); the resume check will flag it", flush=True)
                    else:
                        print(f"[runner] interrupted {sr.name}/{r.dir.name} on purpose (rc={rc}); it must resume from its checkpoint", flush=True)
                        sr.pending.appendleft((r.dir, "normal"))
                elif rc != 0 or st != "succeeded":
                    if st == "running":  # killed without the chance to write its status (OOM killer, SIGKILL, node loss)
                        atomic_write_json(r.dir / "status.json", {"state": "failed", "job_id": r.dir.name, "time": utc_now(), "error": f"process exited with rc={rc} without writing a status"})
                    sr.attempts[r.dir] = sr.attempts.get(r.dir, 0) + 1
                    if sr.attempts[r.dir] <= max_retries:
                        print(f"[runner] RETRY {sr.name}/{r.dir.name} rc={rc} (attempt {sr.attempts[r.dir] + 1})", flush=True)
                        sr.pending.append((r.dir, "normal"))
                    else:
                        sr.failures += 1
                        print(f"[runner] FAILED {sr.name}/{r.dir.name} rc={rc} state={st} (see {r.dir / 'log.txt'})", flush=True)
                        if not keep_going:
                            stop_launching = True
                else:
                    print(f"[runner] finished {sr.name}/{r.dir.name}", flush=True)
                if sr.done():
                    finish_stage(sr)
            running = still
            refresh_rows()
        if report_thread is not None:
            report_thread.join()
        if report_requested and not report_failed and not stop_launching:
            refresh_report()  # the last validated stage(s) arrived while a refresh was running
            if report_thread is not None:
                report_thread.join()
        if report_failed:
            overall_ok = False
        refresh_rows()
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            lock_fh.close()
        except Exception:
            pass
    if overall_ok and all(stages[n].state in ("validated", "skipped") for n in order):
        print("[runner] campaign complete: every stage validated", flush=True)
        return 0
    failed = [n for n in order if stages[n].state == "failed"]
    blocked_s = [n for n in order if stages[n].state == "blocked"]
    print(f"[runner] campaign NOT complete: failed={failed} blocked={blocked_s}", flush=True)
    return 1


def status_report(root: str, details: bool = False) -> str:
    rootp = Path(root)
    lines = [f"campaign: {rootp}"]
    stages_dir = rootp / "stages"
    if not stages_dir.exists():
        return "no stages directory found"
    for sd in sorted(stages_dir.iterdir()):
        jobs = sorted((sd / "jobs").iterdir()) if (sd / "jobs").exists() else []
        counts = stage_counts(jobs)
        lines.append(f"  {sd.name}: total={len(jobs)} " + " ".join(f"{k}={v}" for k, v in counts.items()))
        if details:
            for jd in jobs:
                st = job_state(jd)
                last = ""
                ev = jd / "events.jsonl"
                if ev.exists():
                    try:
                        with ev.open("rb") as f:
                            f.seek(0, os.SEEK_END)
                            size = f.tell()
                            f.seek(max(0, size - 4000))
                            tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                            last = tail[-1][:160] if tail else ""
                    except Exception:
                        pass
                lines.append(f"     [{st:9s}] {jd.name}  {last}")
    return "\n".join(lines)
