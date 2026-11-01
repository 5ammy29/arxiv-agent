CREATE SEQUENCE IF NOT EXISTS paper_processing_fencing_token_seq AS BIGINT;

ALTER TABLE paper_processing_locks
    ADD COLUMN IF NOT EXISTS fencing_token BIGINT;

ALTER TABLE paper_processing_locks
    ALTER COLUMN fencing_token
        SET DEFAULT nextval('paper_processing_fencing_token_seq');

UPDATE paper_processing_locks
SET fencing_token = nextval('paper_processing_fencing_token_seq')
WHERE fencing_token IS NULL;

ALTER TABLE paper_processing_locks
    ALTER COLUMN fencing_token SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_processing_locks_fencing_token
    ON paper_processing_locks (fencing_token);
