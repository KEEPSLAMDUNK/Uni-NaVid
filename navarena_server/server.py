"""NavArena2 WebSocket model server backed by Uni-NaVid.

Wraps the :class:`UniNaVid_Agent` defined in ``offline_eval_uninavid.py``
behind the ``navarena_server.NavigationModelServer`` interface so the
Uni-NaVid VLA can be driven by NavArena2's evaluation bench.

Scope
-----
Only **ObjectNav** is supported right now: the bench's ``goal_category``
is spliced into Uni-NaVid's standard navigation prompt template.

Action mapping
--------------
Each ``predict`` call:

* runs Uni-NaVid once (one new RGB frame is appended to the agent's
  online cache);
* parses the model's textual action list (``forward / left / right /
  stop``) into NavArena waypoint primitives;
* emits a single ``{"waypoints": [...]}`` (or ``{"stop": True}``) action
  message — multiple predicted actions are concatenated into one
  ``waypoints`` list, which the bench's ``WaypointActionAdapter`` executes
  in sequence.

The per-action magnitudes match ``offline_eval_uninavid.py`` (with the
``forward`` step shortened to 0.25 m to match the standard VLN-CE
discretisation used by NavArena):

==========  ==============================
``forward`` ``{x: 0.25, y: 0.0, yaw: 0.0}``
``left``    ``{x: 0.0,  y: 0.0, yaw: -π/6}``
``right``   ``{x: 0.0,  y: 0.0, yaw: +π/6}``
``stop``    ``{"stop": True}``
==========  ==============================

Image format
------------
NavArena ships RGB ``uint8`` HxWx3 arrays per camera. ``offline_eval_uninavid.py``
loads frames via ``cv2.imread`` (BGR) and feeds them straight into the
image processor, so for bit-for-bit consistency with the offline eval
we convert ``RGB -> BGR`` before handing the frame to Uni-NaVid.

Layout note
-----------
This folder is intentionally **not** a Python package (no ``__init__.py``)
so its name does not shadow the installed ``navarena_server`` SDK from
NavArena2. Run the server with::

    python navarena_server/server.py --port 8000

from the Uni-NaVid project root (see ``__main__.py`` for the equivalent
entry point and CLI flags).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
#
# We want two unrelated imports to succeed simultaneously:
#
# 1. ``from offline_eval_uninavid import UniNaVid_Agent`` — this script
#    lives at ``<uninavid_root>/offline_eval_uninavid.py``.
# 2. ``from navarena_server.server.base import NavigationModelServer`` —
#    this comes from the ``navarena-server`` package shipped by NavArena2
#    (installed into site-packages).
#
# The folder containing this file is also called ``navarena_server``,
# which would normally shadow #2. We sidestep that by:
#
# * keeping this folder a PEP 420 namespace candidate (no ``__init__.py``),
#   which means a regular installed package with the same name still wins
#   the import resolution;
# * appending the Uni-NaVid root to ``sys.path`` (using ``append`` so we
#   don't push our local namespace candidate ahead of site-packages).

_HERE = os.path.dirname(os.path.abspath(__file__))
_UNINAVID_ROOT = os.path.dirname(_HERE)


def _scrub_local_navarena_from_syspath() -> None:
    """Drop paths that expose this folder as the ``navarena_server`` namespace.

    ``python navarena_server/server.py`` puts this directory on ``sys.path[0]``,
    which makes ``import navarena_server.server`` resolve back to this file
    instead of the installed NavArena2 SDK (circular import).
    """
    blocked = {os.path.abspath(_HERE), os.path.abspath(_UNINAVID_ROOT)}
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry or os.path.abspath(entry) not in blocked
    ]


_scrub_local_navarena_from_syspath()

from navarena_server.server import NavigationModelServer, SessionContext  # noqa: E402
from navarena_server.server.serve import serve  # noqa: E402

if _UNINAVID_ROOT not in sys.path:
    sys.path.append(_UNINAVID_ROOT)

from offline_eval_uninavid import UniNaVid_Agent  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
DEFAULT_FORWARD_STEP = 0.25  # meters
DEFAULT_TURN_ANGLE_DEG = 30.0  # degrees per left/right step

OBJECTNAV_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been "
    "given a video of historical observations and an image of the current "
    "observation <image>. Your assigned task is: '{instruction}'. Analyze "
    "this series of images to determine your next four actions. The "
    "predicted action should be one of the following: forward, left, right, "
    "or stop."
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class UniNaVidNavArenaServer(NavigationModelServer):
    """NavArena2 model server that delegates inference to :class:`UniNaVid_Agent`.

    Parameters
    ----------
    model_path:
        Filesystem path to the Uni-NaVid checkpoint directory.
    camera_name:
        Optional NavArena camera name to read from ``observation["rgb"]``.
        When ``None`` the first camera in the dict is used; falling back
        keeps the server compatible with any single-camera bench config.
    forward_step:
        Forward translation per ``forward`` action, in meters.
    turn_angle_deg:
        Yaw delta per ``left`` / ``right`` action, in degrees.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        camera_name: str | None = None,
        forward_step: float = DEFAULT_FORWARD_STEP,
        turn_angle_deg: float = DEFAULT_TURN_ANGLE_DEG,
    ) -> None:
        self._camera_name = camera_name
        self._forward_step = float(forward_step)
        self._turn_angle_rad = math.radians(float(turn_angle_deg))
        self._cached_goal_category: str | None = None

        logger.info("Loading Uni-NaVid from %s", model_path)
        self.agent = UniNaVid_Agent(model_path)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def on_episode_start(
        self,
        payload: dict[str, Any],
        ctx: SessionContext,
    ) -> None:
        task = ctx.task or {}
        goal_category = task.get("goal_category")
        if goal_category is None:
            for goal in task.get("goals", []) or []:
                if isinstance(goal, dict) and goal.get("object_category"):
                    goal_category = goal["object_category"]
                    break
        self._cached_goal_category = goal_category

        logger.info(
            "[Episode %s] task_type=%s goal_category=%s",
            ctx.episode_id,
            task.get("task_type", "unknown"),
            goal_category,
        )
        self.agent.reset()

    async def on_episode_end(
        self,
        result: dict[str, Any],
        ctx: SessionContext,
    ) -> None:
        logger.info(
            "[Episode %s] done success=%s after %d steps",
            ctx.episode_id,
            result.get("success"),
            ctx.step,
        )
        self._cached_goal_category = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def predict(
        self,
        observation: dict[str, Any],
        ctx: SessionContext,
    ) -> dict[str, Any]:
        rgb = self._pick_rgb_frame(observation)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        goal_category = observation.get("goal_category") or self._cached_goal_category
        if not goal_category:
            raise ValueError(
                "UniNaVidNavArenaServer currently only supports ObjectNav; "
                "neither observation['goal_category'] nor ctx.task['goal_category'] is set."
            )
        instruction = self._build_instruction(str(goal_category))

        result = self.agent.act({"instruction": instruction, "observations": bgr})
        actions = result.get("actions", [])
        return self._actions_to_navarena(actions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_rgb_frame(self, observation: dict[str, Any]) -> np.ndarray:
        rgb_dict = observation.get("rgb")
        if not isinstance(rgb_dict, dict) or not rgb_dict:
            raise ValueError(
                "Observation is missing the 'rgb' camera dict; "
                "expected {camera_name: HxWx3 uint8 ndarray}."
            )

        if self._camera_name is not None:
            if self._camera_name not in rgb_dict:
                raise KeyError(
                    f"Camera {self._camera_name!r} not found in observation. "
                    f"Available cameras: {list(rgb_dict.keys())}"
                )
            frame = rgb_dict[self._camera_name]
        else:
            frame = next(iter(rgb_dict.values()))

        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected RGB frame shape HxWx3, got shape {tuple(frame.shape)}."
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return frame

    @staticmethod
    def _build_instruction(goal_category: str) -> str:
        # ``goal_category`` from NavArena's ObjectNav evaluator is typically
        # a single bare noun (e.g. "chair", "tv", "bed"); Uni-NaVid's
        # ObjectNav training data uses the natural-language form
        # ``find a/an <category>``.
        category = goal_category.strip()
        article = "an" if category[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        instruction = f"Find {article} {category}."
        return OBJECTNAV_PROMPT_TEMPLATE.format(instruction=instruction)

    def _actions_to_navarena(self, actions: list[str]) -> dict[str, Any]:
        if not actions:
            raise ValueError("Uni-NaVid produced no actions for this step.")

        waypoints: list[dict[str, float]] = []
        for raw in actions:
            action = str(raw).strip().lower()
            if action == "stop":
                if waypoints:
                    logger.debug(
                        "Stop after %d waypoint(s); emitting stop and dropping queued waypoints.",
                        len(waypoints),
                    )
                return {"stop": True}
            if action == "forward":
                waypoints.append(
                    {"x": self._forward_step, "y": 0.0, "yaw": 0.0}
                )
            elif action == "left":
                waypoints.append(
                    {"x": 0.0, "y": 0.0, "yaw": -self._turn_angle_rad}
                )
            elif action == "right":
                waypoints.append(
                    {"x": 0.0, "y": 0.0, "yaw": self._turn_angle_rad}
                )
            else:
                logger.warning("Ignoring unknown Uni-NaVid action: %r", raw)

        if not waypoints:
            raise ValueError(
                f"Uni-NaVid output contained no actionable tokens: {actions!r}"
            )
        return {"waypoints": waypoints}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NavArena2 WebSocket model server for Uni-NaVid (ObjectNav only). "
            "Start this, then point a NavArena bench objectnav_eval.yaml at "
            "ws://<host>:<port>."
        )
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Uni-NaVid checkpoint dir (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--camera-name",
        default=None,
        help=(
            "NavArena camera name to read from observation['rgb']. "
            "If omitted, the first camera in the dict is used."
        ),
    )
    parser.add_argument(
        "--forward-step",
        type=float,
        default=DEFAULT_FORWARD_STEP,
        help=f"Forward translation per step in meters (default: {DEFAULT_FORWARD_STEP}).",
    )
    parser.add_argument(
        "--turn-angle-deg",
        type=float,
        default=DEFAULT_TURN_ANGLE_DEG,
        help=f"Yaw delta per turn in degrees (default: {DEFAULT_TURN_ANGLE_DEG}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = UniNaVidNavArenaServer(
        args.model_path,
        camera_name=args.camera_name,
        forward_step=args.forward_step,
        turn_angle_deg=args.turn_angle_deg,
    )

    logger.info(
        "UniNaVid NavArena server ready: ws://%s:%d (action_space=waypoint, eval_type=objectnav)",
        args.host,
        args.port,
    )
    serve(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
