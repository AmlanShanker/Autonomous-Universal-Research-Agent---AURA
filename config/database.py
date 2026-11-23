import os


def database_uri_from_environment() -> str | None:
    """Return the configured persistent database URI, preferring the explicit setting."""
    return (
        os.getenv("AURA_DATABASE_URI")
        or os.getenv("MONGO_URL")
        or os.getenv("MONGODB_URI")
    )
