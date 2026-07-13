CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    meta_data TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);