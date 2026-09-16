"""Desktop application layer: FastAPI server + pywebview shell + runtime paths.

This package turns the report pipeline into a standalone, chat-style desktop app.
Nothing in the invariant report/ engine imports from here at module scope — the app
depends on the engine, never the reverse. (One deliberate exception:
report/search_cache.py lazily resolves runtime.user_data_dir() at call time, which
keeps engine imports cheap and test-redirectable.)
"""
