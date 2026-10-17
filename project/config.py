import os
from pathlib import Path
from dotenv import load_dotenv


root = Path(__file__).resolve().parent.parent
env_file = root / ".env"

load_dotenv(dotenv_path=env_file)


def get_env_var(var_name: str) -> str:
    name = os.getenv(var_name)
    if not name:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return name


def get_database_url() -> str:
    return get_env_var("DATABASE_URL")


def get_supabase_url() -> str:
    return get_env_var("SUPABASE_URL")


def get_supabase_secret_key() -> str:
    return get_env_var("SUPABASE_SECRET_KEY")


def get_storage_bucket() -> str:
    return os.getenv("SUPABASE_STORAGE_BUCKET", "storage")
