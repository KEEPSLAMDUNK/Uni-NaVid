# Uni-NaVid NavArena Server

WebSocket model server that lets [NavArena2](https://github.com/EI-Nav/NavArena)'s
evaluation bench drive Uni-NaVid for **ObjectNav** tasks.

The server wraps `UniNaVid_Agent` (from `offline_eval_uninavid.py`) behind
NavArena2's `NavigationModelServer` SDK and translates the bench's
`goal_category` observation into Uni-NaVid's standard ObjectNav prompt.

## Layout

```
navarena_server/
├── README.md       # this file
├── server.py       # UniNaVidNavArenaServer + CLI entry (main())
└── __main__.py     # thin CLI wrapper (does NOT enable `python -m navarena_server`)
```

> **Why no `__init__.py`?** This folder shares its name with the
> `navarena_server` Python package shipped by NavArena2. We intentionally
> keep this folder as a PEP 420 namespace candidate (no `__init__.py`)
> so the installed regular package wins import resolution. As a result
> `python -m navarena_server` is **not** supported here; use the script
> invocations below.

## Prerequisites

Install both packages into the same Python environment:

```bash
# 1) Uni-NaVid (see Uni-NaVid/README.md for the full install incl. flash-attn)
cd /path/to/Uni-NaVid
pip install -e .

# 2) NavArena2's navarena-server SDK
cd /path/to/NavArena2
uv sync --frozen --package navarena-server   # or: pip install -e packages/navarena-server
```

Download the Uni-NaVid checkpoint (e.g. `uninavid-7b-full-224-video-fps-1-grid-2`)
into `Uni-NaVid/model_zoo/` as documented in the project README.

## Start the server

From the **Uni-NaVid project root**:

```bash
python navarena_server/server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model-path model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
```

Equivalent:

```bash
python navarena_server/__main__.py --port 8000
```

Useful flags (`--help` for the full list):

| Flag                | Default                                          | Meaning                                      |
| ------------------- | ------------------------------------------------ | -------------------------------------------- |
| `--host`            | `0.0.0.0`                                        | Bind host                                    |
| `--port`            | `8000`                                           | Bind port                                    |
| `--model-path`      | `model_zoo/uninavid-7b-full-224-video-fps-1-grid-2` | Uni-NaVid checkpoint directory               |
| `--camera-name`     | `None` (first camera in observation)             | NavArena camera name to read RGB from        |
| `--forward-step`    | `0.25`                                           | Forward translation per `forward` action (m) |
| `--turn-angle-deg`  | `30.0`                                           | Yaw delta per `left`/`right` action (deg)    |
| `--log-level`       | `INFO`                                           | Python logging level                         |

## Configure the bench

In NavArena2, copy `packages/navarena-bench/configs/eval/objectnav_eval.yaml`
and change `server.url` to point at this process:

```yaml
eval_type: "objectnav"

server:
  url: "ws://<server-host>:8000"
  timeout: 30.0
  image_format: "jpeg"

eval_settings:
  action_space: "waypoint"   # required: server returns {waypoints: [...]} / {stop: true}
  num_episodes: 100
  batch_size: 1              # batch mode is not implemented here yet
  output_path: "./eval_results"
  max_steps_per_episode: 500
```

Then launch the bench as you would for any other agent.

## Behaviour

- `on_episode_start`: caches `task.goal_category` (or the first
  `goals[].object_category` it finds) and resets the Uni-NaVid online
  feature cache.
- `predict`: every step
  - reads one RGB frame from `observation["rgb"]` (configurable camera),
  - converts `RGB -> BGR` to match `offline_eval_uninavid.py`,
  - splices the goal category into Uni-NaVid's ObjectNav prompt
    (`"Find a/an {category}."`),
  - runs `UniNaVid_Agent.act(...)` once,
  - maps the model's textual action list to NavArena waypoint primitives:

    | Uni-NaVid output | NavArena action                                  |
    | ---------------- | ------------------------------------------------ |
    | `forward`        | waypoint `{x: 0.25, y: 0.0, yaw: 0.0}`           |
    | `left`           | waypoint `{x: 0.0,  y: 0.0, yaw: -π/6}`          |
    | `right`          | waypoint `{x: 0.0,  y: 0.0, yaw: +π/6}`          |
    | `stop`           | `{"stop": true}` (drops any queued waypoints)    |

  - returns a single `{"waypoints": [...]}` message containing all
    forward/left/right primitives the model planned, or `{"stop": true}`
    as soon as the model emits `stop`. The bench's `WaypointActionAdapter`
    executes the queued waypoints sequentially.

## Limitations

- ObjectNav only. PointNav / ImageNav / VLN are not wired up.
- Batch mode (`batch_size > 1`) is not implemented; `batch_predict`
  falls back to per-slot `predict` calls inherited from the base class.
- Image colour conversion matches `offline_eval_uninavid.py` (BGR input
  to the model). If you confirm the model's image processor handles
  channel order itself, the `cv2.cvtColor` call in `predict` can be
  removed.
