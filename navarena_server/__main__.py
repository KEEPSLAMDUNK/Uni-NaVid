"""CLI entry point for the Uni-NaVid NavArena server.

Mirrors ``server.py``'s ``main()`` so the launcher can be invoked either
as a direct script::

    python navarena_server/__main__.py --port 8000

or via the explicit script file::

    python navarena_server/server.py --port 8000

Note: ``python -m navarena_server`` is intentionally NOT supported here,
because this folder is not a regular Python package (no
``__init__.py``) — that's by design so its name does not shadow the
installed ``navarena_server`` SDK shipped by NavArena2.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from server import main  # type: ignore[import-not-found]  # noqa: E402

if __name__ == "__main__":
    main()
