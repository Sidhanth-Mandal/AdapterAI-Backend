CREATE TABLE Users (
    user_id VARCHAR PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Templates (
    template_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    behaviour_prompt TEXT,
    tool_generation_prompt TEXT,
    tool_information TEXT,
    created_by VARCHAR NOT NULL REFERENCES Users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Conversations (
    conv_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL REFERENCES Users(user_id),
    template_id VARCHAR NOT NULL REFERENCES Templates(template_id),
    title VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversation_memory (
    conversation_id UUID PRIMARY KEY,
    summary TEXT,
    last_summarized_message_id UUID,
    unsummarized_token_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE Messages (
    message_id VARCHAR PRIMARY KEY,
    conv_id VARCHAR NOT NULL REFERENCES Conversations(conv_id),
    role VARCHAR NOT NULL, -- 'system/user/assistant/tool'
    content TEXT NOT NULL,
    token_count INT,
    sequence_number INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Tools (
    tool_id VARCHAR PRIMARY KEY,
    template_id VARCHAR NOT NULL REFERENCES Templates(template_id),
    name VARCHAR NOT NULL,
    description TEXT,
    language VARCHAR, -- 'python/js/etc'
    tool_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version VARCHAR
);

CREATE TABLE Attachments (
    attachment_id VARCHAR PRIMARY KEY,
    message_id VARCHAR NOT NULL REFERENCES Messages(message_id),
    file_name VARCHAR NOT NULL,
    mime_type VARCHAR,
    storage_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE TEMP_MESSAGES (
    message_id      VARCHAR PRIMARY KEY,
    template_id     VARCHAR NOT NULL,
    role            VARCHAR NOT NULL, -- system/user/assistant
    content         TEXT NOT NULL,
    token_count     INT,
    sequence_number INT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_temp_messages_template_id ON TEMP_MESSAGES (template_id);
