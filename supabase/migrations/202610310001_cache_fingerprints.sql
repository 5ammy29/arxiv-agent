ALTER TABLE public.paper_artifacts
    ADD COLUMN IF NOT EXISTS content_fingerprint TEXT,
    ADD COLUMN IF NOT EXISTS chunking_fingerprint TEXT;

ALTER TABLE public.paper_vector_indexes
    ADD COLUMN IF NOT EXISTS content_fingerprint TEXT,
    ADD COLUMN IF NOT EXISTS chunking_fingerprint TEXT;

-- Existing rows deliberately remain NULL. Application code treats those
-- pre-fingerprint cache entries as untrusted and rebuilds them.
CREATE INDEX IF NOT EXISTS idx_paper_artifacts_content_fingerprint
    ON public.paper_artifacts (content_fingerprint);
