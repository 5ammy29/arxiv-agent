CREATE TABLE paper_artifacts (
    paper_id UUID PRIMARY KEY
        REFERENCES papers(id) ON DELETE CASCADE,
    pdf_storage_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE paper_pages (
    paper_id UUID NOT NULL
        REFERENCES papers(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    text TEXT NOT NULL,
    PRIMARY KEY (paper_id, page_number)
);

CREATE TABLE paper_chunks (
    paper_id UUID NOT NULL,
    chunk_id INTEGER NOT NULL CHECK (chunk_id >= 0),
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    text TEXT NOT NULL,
    PRIMARY KEY (paper_id, chunk_id),
    FOREIGN KEY (paper_id, page_number)
        REFERENCES paper_pages(paper_id, page_number)
        ON DELETE CASCADE
);

CREATE INDEX idx_paper_chunks_page
    ON paper_chunks (paper_id, page_number);
