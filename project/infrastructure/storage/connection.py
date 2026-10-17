from supabase import Client, create_client

from project.config import (
    get_supabase_url,
    get_supabase_secret_key,
    get_storage_bucket
)


storage_client: Client | None = None


def get_storage_client() -> Client:
    global storage_client

    if storage_client is None:
        supabase_url = get_supabase_url()
        secret_key = get_supabase_secret_key()

        storage_client = create_client(supabase_url, secret_key)

    return storage_client


def get_storage_bucket_client():
    client = get_storage_client()
    bucket_name = get_storage_bucket()

    return client.storage.from_(bucket_name)
