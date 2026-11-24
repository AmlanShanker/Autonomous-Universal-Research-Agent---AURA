"""Legacy settings compatibility.

Database configuration is resolved by config.database at application startup.
Importing this module must not require a database URI.
"""

import os

from dotenv import load_dotenv

load_dotenv()

MONGO_URL: str | None = os.getenv("MONGO_URL")
