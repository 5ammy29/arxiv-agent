CREATE TABLE paper_processing_locks (
    paper_id UUID PRIMARY KEY
        REFERENCES papers(id) ON DELETE CASCADE,

    owner_token UUID NOT NULL,

    acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    lease_expires_at TIMESTAMPTZ NOT NULL,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_lease_expiry
        CHECK (lease_expires_at > acquired_at)
);

CREATE INDEX idx_paper_processing_locks_expiry
    ON paper_processing_locks (lease_expires_at);