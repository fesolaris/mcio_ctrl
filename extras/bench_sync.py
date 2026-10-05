"""
Reproducible SYNC-mode throughput benchmark with an optional cProfile breakdown.

Two scenarios, both launched hidden at the training frame size:

  single  one SYNC client in a singleplayer world (ticks/s == steps/s)
  duel    host + guest clients in one LAN world driven by MCioMultiEnv, frame_skip ticks
          per decision, like a self-play actor pair (decisions/s per client)

Each scenario runs --warmup steps and then --repeats measured windows of --steps steps.
One JSON line per scenario is printed so runs can be compared (see --label).

    python extras/bench_sync.py single --instance AITestInstance --world Arenas
    python extras/bench_sync.py duel --instance AIInstance7 --guest AIInstance8 --world Arenas
    python extras/bench_sync.py single --instance AITestInstance --profile

Launch tuning without touching the library: --jvm-arg (repeatable) and --env KEY=VALUE.
Everything this script launches is closed on exit.
"""

import argparse
import cProfile
import itertools
import json
import logging
import pstats
import statistics
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import mcio_ctrl as mcio
from mcio_ctrl.envs import mcio_env
from mcio_ctrl.envs.multi_env import MCioMultiEnv
from mcio_ctrl.types import GameMode, MCioMode, RunOptions

LOG = logging.getLogger("bench_sync")

SETUP_COMMANDS = [
    "difficulty peaceful",
    "gamerule doMobSpawning false",
    "gamerule sendCommandFeedback false",
    "gamerule doDaylightCycle false",
    "time set day",
]


def action_for(env: mcio_env.MCioEnv, move: bool) -> dict[str, Any]:
    action = env.get_noop_action()
    if move:
        action["cursor_delta"] = np.array((3, 1), dtype=np.int32)
        action["W"] = np.int64(1)
    return action


def step_options(args: argparse.Namespace, index: int) -> dict[str, Any]:
    """Options for the index-th step: ask for no frame except every --frame-every-th step."""
    if args.frame_every <= 1:
        return {}
    if (index + 1) % args.frame_every == 0:
        return {}
    return {"send_frame": False}


def measure(
    step: Callable[[], None], steps: int, warmup: int, repeats: int, profile: bool
) -> tuple[list[float], str | None]:
    for _ in range(warmup):
        step()
    rates: list[float] = []
    prof = cProfile.Profile() if profile else None
    for _ in range(repeats):
        start = time.perf_counter()
        if prof is not None:
            prof.enable()
        for _ in range(steps):
            step()
        if prof is not None:
            prof.disable()
        rates.append(steps / (time.perf_counter() - start))
    report = None
    if prof is not None:
        import io

        out = io.StringIO()
        stats = pstats.Stats(prof, stream=out).strip_dirs()
        stats.sort_stats("tottime").print_stats(22)
        report = out.getvalue()
    return rates, report


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    env_extra = dict(kv.split("=", 1) for kv in args.env)
    launch_opts = RunOptions(
        instance_name=args.instance,
        world_name=args.world,
        width=args.size,
        height=args.size,
        mcio_mode=MCioMode.SYNC,
        hide_window=True,
        action_port=args.port,
        observation_port=args.port + 4000,
        cleanup_on_signal=False,
        env_extra=env_extra,
    )
    env = mcio_env.MCioEnv(launch_opts)
    try:
        env.reset(options={"commands": SETUP_COMMANDS + ["gamemode creative"]})
        env.skip_steps(30)
        action = action_for(env, args.move)
        counter = itertools.count()

        def step_once() -> None:
            i = next(counter)
            env.begin_step(action, step_options(args, i))
            env.end_step(action)

        rates, report = measure(step_once, args.steps, args.warmup, args.repeats, args.profile)
        last = env.last_frame
        return {
            "scenario": "single",
            "unit": "steps/s",
            "rates": rates,
            "frame_mean": float(last.mean()) if last is not None else None,
            "frame_shape": list(last.shape) if last is not None else None,
            "profile": report,
        }
    finally:
        env.close()


def run_duel(args: argparse.Namespace) -> dict[str, Any]:
    env_extra = dict(kv.split("=", 1) for kv in args.env)
    ap, op, lan = args.port, args.port + 4000, args.lan_port
    host_opts = RunOptions(
        instance_name=args.instance,
        world_name=args.world,
        width=args.size,
        height=args.size,
        mcio_mode=MCioMode.SYNC,
        hide_window=True,
        open_to_lan=True,
        open_to_lan_port=lan,
        open_to_lan_mode=GameMode.SURVIVAL,
        mc_username="BenchH",
        action_port=ap,
        observation_port=op,
        cleanup_on_signal=False,
        env_extra=env_extra,
    )
    guest_opts = RunOptions(
        instance_name=args.guest,
        mc_username="BenchG",
        width=args.size,
        height=args.size,
        mcio_mode=MCioMode.SYNC,
        hide_window=True,
        action_port=ap + 1,
        observation_port=op + 1,
        cleanup_on_signal=False,
        env_extra=env_extra,
    )
    guest_connect = RunOptions(
        width=args.size,
        height=args.size,
        mcio_mode=MCioMode.SYNC,
        action_port=ap + 1,
        observation_port=op + 1,
    )
    guest_launcher = mcio.instance.Launcher(guest_opts)
    host = mcio_env.MCioEnv(host_opts)
    guest = mcio_env.MCioEnv(guest_connect)
    multi = MCioMultiEnv([host, guest])
    try:
        host.reset(options={"commands": SETUP_COMMANDS})
        stop = threading.Event()
        noop = host.get_noop_action()

        def pump() -> None:
            while not stop.is_set():
                host.step(noop)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        guest_launcher.mll_opts["quickPlayMultiplayer"] = f"localhost:{lan}"
        guest_launcher.launch(wait=False)
        guest.reset()
        stop.set()
        guest.send_noop()
        guest.recv_observation()
        thread.join()

        actions = [action_for(host, args.move), action_for(guest, args.move)]

        def decision() -> None:
            for tick in range(args.frame_skip):
                last_tick = tick == args.frame_skip - 1
                opts: list[dict[str, Any]] = [
                    {} if (last_tick or not args.skip_frames) else {"send_frame": False}
                    for _ in range(2)
                ]
                multi.step(actions, opts)

        rates, report = measure(
            decision, args.steps, args.warmup, args.repeats, args.profile
        )
        last = host.last_frame
        return {
            "scenario": "duel",
            "unit": "decisions/s/client",
            "frame_skip": args.frame_skip,
            "rates": rates,
            "frame_mean": float(last.mean()) if last is not None else None,
            "frame_shape": list(last.shape) if last is not None else None,
            "profile": report,
        }
    finally:
        multi.close()
        guest_launcher.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scenario", choices=["single", "duel"])
    parser.add_argument("--instance", required=True, help="Instance to launch (duel: the LAN host)")
    parser.add_argument("--guest", help="duel: instance for the guest client")
    parser.add_argument("--world", default="Arenas")
    parser.add_argument("--size", type=int, default=256, help="Frame width and height")
    parser.add_argument("--steps", type=int, default=600, help="Steps (duel: decisions) per window")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3, help="Measured windows")
    parser.add_argument("--frame-skip", type=int, default=2, help="duel: ticks per decision")
    parser.add_argument("--port", type=int, default=4101, help="Action port; observation is +4000")
    parser.add_argument("--lan-port", type=int, default=25601)
    parser.add_argument("--jvm-arg", action="append", default=[], help="Extra JVM argument (repeatable)")
    parser.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE for Minecraft (repeatable)")
    parser.add_argument("--move", action="store_true", help="Send input events and mouse movement")
    parser.add_argument(
        "--frame-every",
        type=int,
        default=1,
        help="single: request the frame only every Nth step (others send send_frame=False)",
    )
    parser.add_argument(
        "--skip-frames",
        action="store_true",
        help="duel: the non-final ticks of each decision send send_frame=False (action-repeat pattern)",
    )
    parser.add_argument("--profile", action="store_true", help="cProfile the measured windows")
    parser.add_argument("--label", default="", help="Free-form tag copied into the JSON line")
    args = parser.parse_args()
    if args.scenario == "duel" and not args.guest:
        parser.error("duel needs --guest")
    return args


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    result = run_single(args) if args.scenario == "single" else run_duel(args)
    report = result.pop("profile")
    rates = result["rates"]
    result["median"] = statistics.median(rates)
    result["label"] = args.label
    result["instance"] = args.instance
    result["size"] = args.size
    result["jvm_args"] = args.jvm_arg
    print("BENCH " + json.dumps(result))
    if report:
        print(report)


if __name__ == "__main__":
    main()
