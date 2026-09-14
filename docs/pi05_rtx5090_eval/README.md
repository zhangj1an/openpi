# π0.5 LIBERO-Spatial eval artifacts (RTX 5090)

Raw results behind `docs/pi05_rtx5090_latency.md`. Hardware, client protocol and checkpoints are described there.

## `libero_spatial/` — closed-loop rollouts (10 tasks × 10 episodes)

| File | Success | Per task | Round trip p50 / p90 / p99 (ms) | Server infer p50 (ms) | Mean steps |
| --- | ---: | --- | ---: | ---: | ---: |
| `branch_pytorch_nvfp4.json` | 100/100 | 10 10 10 10 10 10 10 10 10 10 | 25.1 / 25.5 / 26.1 | 24.2 | 112.1 |
| `upstream_openpi_jax.json` | 100/100 | 10 10 10 10 10 10 10 10 10 10 | 73.1 / 75.4 / 77.4 | 72.2 | 112.0 |
| `vllm_omni_pi05_cudagraph_lerobot_ckpt.json` | 97/100 | 10 10 10 10 10 7 10 10 10 10 | 94.6 / 97.2 / 101.6 | – | 115.4 |
| `vllm_omni_pi05_cudagraph_openpi_ckpt.json` | 99/100 | 10 10 10 10 10 9 10 10 10 10 | 93.3 / 95.8 / 99.7 | – | 113.6 |

Each file has per-episode results (`episodes`), latency summaries, and every chunk's round trip (`raw`).
The vLLM-Omni rows are for reference (branch `pi05-cudagraph` of vllm-omni, see the main doc).

## `latency/` — in-process `Policy.infer` harness

One real LIBERO frame, fixed noise. `lat_*`, `flags_*`, `jax_*`, `branch_lat_*`: JAX latency with XLA autotuning;
`exact_*`, `branch_exact_*`: JAX actions with autotuning disabled, `maxabs_vs_baseline` against upstream `main`;
`torch_*`: PyTorch path, `maxabs_vs_jax_ref` against openpi JAX with identical noise.

## `flash/`

FLASH draft training on LIBERO-Spatial (1× H100, 100 epochs, `--cache-prefixes --confidence`, bf16 teacher):
`metrics.jsonl` has the validation metrics every 500 steps, `draft_meta.json` those of the best checkpoint (step
37,000). The checkpoint itself is not in the repository. Findings: `docs/pi05_rtx5090_latency.md`.

## `scripts/`

- `sim_client_openpi.py`: LIBERO client (openpi websocket protocol; `--server vllm-omni`, `--flash`). Runs in an environment with LIBERO (e.g. LeRobot's `hf-libero`) and `openpi-client`.
- `run_openpi_libero_eval.sh`, `run_branch_libero_eval.sh`: server + client wrappers used for the rollouts.
- `exp_openpi.py`, `exp_openpi_torch.py`, `openpi_patches.py`: latency / exactness harness and the prototypes tried before implementing changes in `src/`.

Paths inside the scripts (`/dev/shm/...`, `/workspace/...`) are from the benchmark machine; adjust them to your setup.
