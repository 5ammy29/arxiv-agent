CREATE TABLE paper_vector_indexes (
    paper_id UUID PRIMARY KEY
        REFERENCES papers(id) ON DELETE CASCADE,

    index_storage_path TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL
        CHECK (embedding_dimension > 0),

    chunk_ids JSONB NOT NULL,
    chunk_count INTEGER NOT NULL
        CHECK (chunk_count > 0),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CHECK (jsonb_typeof(chunk_ids) = 'array'),
    CHECK (jsonb_array_length(chunk_ids) = chunk_count)
);