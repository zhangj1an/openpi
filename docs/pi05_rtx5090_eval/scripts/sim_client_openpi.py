#!/usr/bin/env python
"""Closed-loop LIBERO rollouts against an openpi websocket policy server (scripts/serve_policy.py --env LIBERO).

Mirrors openpi/examples/libero/main.py (224px resize_with_pad, 180-degree flip, replan every 5 steps) and times
every client.infer() round trip (= e2e latency seen by the robot: msgpack + websocket + server policy.infer).
Runs in the LeRobot venv (hf-libero + openpi-client). Also dumps one real observation for bench_openpi.py.
"""

import argparse
import collections
import json
import math
import os
import pathlib
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as _wcp
import websockets.sync.client as _ws_client


class OpenpiNoPingClient:
    """openpi websocket protocol without keepalive pings: the first request may block the server for minutes
    (PyTorch compile + CUDA graph capture), which would otherwise close the connection."""

    def __init__(self, host, port):
        self._packer = msgpack_numpy.Packer()
        self._ws = _ws_client.connect(f"ws://{host}:{port}", compression=None, max_size=None, ping_interval=None)
        self._metadata = msgpack_numpy.unpackb(self._ws.recv())

    def get_server_metadata(self):
        return self._metadata

    def infer(self, obs):
        self._ws.send(self._packer.pack(obs))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Inference failed: {resp}")
        return msgpack_numpy.unpackb(resp)


class VllmOmniClient:
    """OpenPI wire protocol against vLLM-Omni's /v1/realtime/robot/openpi (reply is a bare action ndarray)."""

    def __init__(self, host, port, path):
        self._uri = f"ws://{host}:{port}{path}"
        self._packer = msgpack_numpy.Packer()
        self._ws = _ws_client.connect(self._uri, compression=None, max_size=None, ping_interval=300, ping_timeout=3600)
        self._metadata = msgpack_numpy.unpackb(self._ws.recv())

    def get_server_metadata(self):
        return self._metadata

    def infer(self, obs):
        obs = dict(obs)
        obs["endpoint"] = "infer"
        self._ws.send(self._packer.pack(obs))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Inference failed: {resp}")
        out = msgpack_numpy.unpackb(resp)
        if isinstance(out, dict) and out.get("type") == "error":
            raise RuntimeError(f"Inference failed: {out}")
        return {"actions": np.asarray(out, dtype=np.float32)}

DUMMY = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server", default="openpi", choices=["openpi", "vllm-omni"])
    p.add_argument("--path", default="/v1/realtime/robot/openpi")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-ids", default="0,1,2,3,4")
    p.add_argument("--episodes-per-task", type=int, default=2)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--videos", type=int, default=2)
    p.add_argument("--dump-obs", default="")
    p.add_argument("--flash", action="store_true", help="FLASH server: send flash_reset per episode, log round kinds")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.server == "vllm-omni":
        client = VllmOmniClient(args.host, args.port, args.path)
        img_key, wrist_key, state_key = "observation.images.image", "observation.images.image2", "state"
    else:
        client = OpenpiNoPingClient(args.host, args.port)
        img_key, wrist_key, state_key = "observation/image", "observation/wrist_image", "observation/state"
    print("server metadata:", client.get_server_metadata(), flush=True)
    dummy = {img_key: np.zeros((224, 224, 3), np.uint8), wrist_key: np.zeros((224, 224, 3), np.uint8),
             state_key: np.zeros(8, np.float32), "prompt": "warmup"}
    warmup_ms = []
    for _ in range(5):  # server jit-compiles on the first request; exclude it from the stats
        t = time.perf_counter()
        client.infer(dummy)
        warmup_ms.append((time.perf_counter() - t) * 1e3)
    print("warmup ms:", [round(w, 1) for w in warmup_ms], flush=True)

    rt, srv, pre = [], [], []
    kinds, accepted_lens = [], []
    episodes, videos = [], 0
    for tid in [int(t) for t in args.task_ids.split(",")]:
        task = suite.get_task(tid)
        init_states = suite.get_task_init_states(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        for ep in range(args.episodes_per_task):
            env.reset()
            obs = env.set_init_state(init_states[ep])
            plan, frames, t, done, n_chunks = collections.deque(), [], 0, False, 0
            first_request = True
            t_ep = time.perf_counter()
            while t < MAX_STEPS[args.suite] + 10:
                if t < 10:
                    obs, _, done, _ = env.step(DUMMY)
                    t += 1
                    continue
                t0 = time.perf_counter()
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                if videos < args.videos:
                    frames.append(np.concatenate([img, wrist], 1))
                if not plan:
                    element = {
                        img_key: img,
                        wrist_key: wrist,
                        state_key: np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ).astype(np.float32),
                        "prompt": str(task.language),
                    }
                    if args.dump_obs and not os.path.exists(args.dump_obs):
                        np.savez(args.dump_obs, image=img, wrist_image=wrist, state=element[state_key],
                                 prompt=element["prompt"])
                    if args.flash and first_request:
                        element["flash_reset"] = True
                    first_request = False
                    t1 = time.perf_counter()
                    resp = client.infer(element)
                    t2 = time.perf_counter()
                    pre.append((t1 - t0) * 1e3)
                    rt.append((t2 - t1) * 1e3)
                    if args.flash:
                        kinds.append(resp.get("flash", {}).get("round", "?"))
                        accepted_lens.append(int(resp.get("accepted_prefix_len", len(resp["actions"]))))
                    if "server_timing" in resp:
                        srv.append(resp["server_timing"]["infer_ms"])
                    n_chunks += 1
                    plan.extend(resp["actions"][: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                if done:
                    break
                t += 1
            ep_info = {"task_id": tid, "task": task.language, "episode": ep, "success": bool(done), "steps": t,
                       "chunks": n_chunks, "wall_s": time.perf_counter() - t_ep}
            episodes.append(ep_info)
            print("episode:", ep_info, flush=True)
            if frames:
                imageio.mimwrite(out / f"video_task{tid}_ep{ep}_{'success' if done else 'fail'}.mp4", frames, fps=20)
                videos += 1
        env.close()

    # first chunk of each episode is already warm (server jit-compiled on its first request), keep all
    res = {
        "framework": f"{args.server} websocket server + LIBERO client",
        "config": vars(args),
        "warmup_ms": warmup_ms,
        "client_preprocess_ms": stats(pre),
        "client_roundtrip_ms": stats(rt),
        "server_policy_infer_ms": stats(srv) if srv else None,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "episodes": episodes,
        "raw": {"client_roundtrip": rt, "server_policy_infer": srv, "client_preprocess": pre, "flash_kind": kinds,
                "accepted_prefix_len": accepted_lens},
    }
    if args.flash and kinds:
        by_kind = {}
        for k, r in zip(kinds, rt):
            by_kind.setdefault(k, []).append(r)
        res["flash"] = {
            "rounds": {k: len(v) for k, v in by_kind.items()},
            "roundtrip_ms_by_kind": {k: stats(v) for k, v in by_kind.items()},
            "flash_round_rate": float(np.mean([k == "flash" for k in kinds])),
            "mean_actions_per_request": float(np.mean(accepted_lens)),
            "ms_per_executed_action": float(np.sum(rt) / max(1, sum(min(a, args.replan_steps) for a in accepted_lens))),
        }
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("SUMMARY", json.dumps({"roundtrip_p50_ms": round(res["client_roundtrip_ms"]["p50"], 2),
                                 "server_infer_p50_ms": round(res["server_policy_infer_ms"]["p50"], 2) if srv else None,
                                 "success_rate": res["success_rate"]}), flush=True)


if __name__ == "__main__":
    main()
