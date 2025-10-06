-- backend/app/db/init/010_schema.sql
-- One row per paper
CREATE TABLE IF NOT EXISTS papers (
  doc_id      TEXT PRIMARY KEY,
  title       TEXT NOT NULL,
  abstract    TEXT NOT NULL,
  embedding   vector(768) NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS corpus_state (
  corpus_name TEXT PRIMARY KEY,
  fingerprint TEXT NOT NULL,
  updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS papers_embedding_ivf
ON papers USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS papers_doc_id_idx ON papers (doc_id);
