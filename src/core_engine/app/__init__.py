"""Desktop application layer: FastAPI server + pywebview shell + runtime paths.

This package turns the report pipeline into a standalone, chat-style desktop app.
Nothing in the invariant report/ engine imports from here — the app depends on the
engine, never the reverse.
"""
