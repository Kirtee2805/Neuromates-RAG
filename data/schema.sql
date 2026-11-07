CREATE TABLE knowledge_chunks (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, category TEXT NOT NULL,
 topic TEXT NOT NULL, content TEXT NOT NULL, keywords TEXT NOT NULL,
 source_status TEXT NOT NULL, language TEXT NOT NULL
);
CREATE INDEX idx_category ON knowledge_chunks(category);
CREATE INDEX idx_topic ON knowledge_chunks(topic);
