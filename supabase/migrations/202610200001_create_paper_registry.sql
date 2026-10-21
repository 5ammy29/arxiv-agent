CREATE TABLE IF NOT EXISTS public.papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    arxiv_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),

    title TEXT NOT NULL,
    authors JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(authors) = 'array'),
    abstract TEXT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    arxiv_updated_at TIMESTAMPTZ NOT NULL,
    url TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    categories JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(categories) = 'array'),

    processing_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (
            processing_status IN (
                'pending',
                'downloading',
                'parsing',
                'ready',
                'failed'
            )
        ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_papers_arxiv_id_version
        UNIQUE (arxiv_id, version)
);

CREATE TABLE IF NOT EXISTS public.session_papers (
    session_id UUID NOT NULL
        REFERENCES public.sessions(id) ON DELETE CASCADE,

    paper_id UUID NOT NULL
        REFERENCES public.papers(id) ON DELETE CASCADE,

    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (session_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_papers_processing_status
    ON public.papers (processing_status);

CREATE INDEX IF NOT EXISTS idx_session_papers_paper_id
    ON public.session_papers (paper_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_session_papers_one_active
    ON public.session_papers (session_id)
    WHERE is_active = TRUE;
