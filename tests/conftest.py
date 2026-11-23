"""Keep developer .env settings from leaking into the pytest process."""

import os


# API security is tested explicitly with monkeypatched credentials in
# tests/test_api_security.py. All other tests should use the default local
# development mode, regardless of the developer machine's .env file.
os.environ["AURA_API_AUTH_ENABLED"] = "false"
os.environ.pop("AURA_API_KEYS", None)
