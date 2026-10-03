"""Dev entrypoint - use this instead of the bare `uvicorn app.main:app` CLI.

On Windows, Python 3.8+ defaults to ProactorEventLoopPolicy, and uvicorn
only overrides that with the WindowsSelectorEventLoopPolicy when it spawns
a subprocess (--reload / --workers>1) - a plain single-process run keeps
Proactor. Proactor's pipe transport has a long-standing bug where serving a
file over a keep-alive connection can reset the socket mid-response
(ConnectionResetError in _ProactorBasePipeTransport._call_connection_lost),
which is exactly what was truncating /api/jobs/{id}/brain.glb and
tumor.glb fetches in the browser. Must be set before uvicorn creates its
event loop, so it has to happen here, not inside app/main.py.
"""

import asyncio
import sys

import uvicorn

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # reload=False: WatchFiles's hot-swap was unreliable on this machine
    # (old worker kept serving after a "Reloading..." log line, silently
    # masking code changes) - restart this script by hand after edits.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
