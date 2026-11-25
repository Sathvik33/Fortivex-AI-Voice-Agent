"""
Database schema definitions for the Autonomous AI Sales Voice Agent.
Uses NeonDB (PostgreSQL) with pgvector extension for RAG embeddings.
"""

CREATE_EXTENSIONS = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
"""

CREATE_CUSTOMERS_TABLE = """
CREATE TABLE IF NOT EXISTS customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(50),
    company VARCHAR(255),
    company_size INTEGER,
    budget DECIMAL(15, 2),
    budget_currency VARCHAR(10) DEFAULT 'INR',
    requirements TEXT,
    lead_score DECIMAL(5, 2) DEFAULT 0,
    lead_category VARCHAR(20) DEFAULT 'COLD',
    sentiment VARCHAR(20) DEFAULT 'neutral',
    preferred_language VARCHAR(20) DEFAULT 'en',
    last_interaction TIMESTAMP WITH TIME ZONE,
    follow_up_status VARCHAR(50) DEFAULT 'pending',
    decision_maker BOOLEAN DEFAULT FALSE,
    purchase_timeline VARCHAR(100),
    objections TEXT,
    buying_intent VARCHAR(20) DEFAULT 'low',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    customer_id UUID REFERENCES customers(id),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    detected_language VARCHAR(20) DEFAULT 'en',
    final_lead_score DECIMAL(5, 2),
    final_sentiment VARCHAR(20),
    summary TEXT,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    detected_language VARCHAR(20),
    sentiment VARCHAR(20),
    intent VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

CREATE_APPOINTMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS appointments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID REFERENCES customers(id),
    conversation_id UUID REFERENCES conversations(id),
    appointment_date DATE NOT NULL,
    appointment_time TIME NOT NULL,
    duration_minutes INTEGER DEFAULT 60,
    purpose VARCHAR(500) DEFAULT 'Product Demonstration',
    status VARCHAR(50) DEFAULT 'confirmed',
    confirmed_by_customer BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT no_overlap UNIQUE (appointment_date, appointment_time)
);
"""

CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    content_hash VARCHAR(64) UNIQUE,
    total_chunks INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'processing',
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);
"""

CREATE_DOCUMENT_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

CREATE_CALL_SUMMARIES_TABLE = """
CREATE TABLE IF NOT EXISTS call_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id),
    customer_id UUID REFERENCES customers(id),
    customer_name VARCHAR(255),
    company VARCHAR(255),
    requirements TEXT,
    budget TEXT,
    purchase_timeline VARCHAR(200),
    product_interests TEXT,
    objections TEXT,
    sentiment VARCHAR(20),
    lead_score DECIMAL(5, 2),
    lead_category VARCHAR(20),
    appointment_booked BOOLEAN DEFAULT FALSE,
    appointment_id UUID REFERENCES appointments(id),
    recommended_action TEXT,
    full_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

CREATE_ANALYTICS_TABLE = """
CREATE TABLE IF NOT EXISTS analytics_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    conversation_id UUID,
    customer_id UUID,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

CREATE_AVAILABLE_SLOTS_TABLE = """
CREATE TABLE IF NOT EXISTS available_slots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slot_date DATE NOT NULL,
    slot_time TIME NOT NULL,
    is_available BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (slot_date, slot_time)
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_lead_category ON customers(lead_category);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_customer ON conversations(customer_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_appointments_date ON appointments(appointment_date, appointment_time);
CREATE INDEX IF NOT EXISTS idx_appointments_customer ON appointments(customer_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_created ON analytics_events(created_at);
"""

CREATE_VECTOR_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

ALL_SCHEMA_SQL = [
    CREATE_EXTENSIONS,
    CREATE_CUSTOMERS_TABLE,
    CREATE_CONVERSATIONS_TABLE,
    CREATE_MESSAGES_TABLE,
    CREATE_APPOINTMENTS_TABLE,
    CREATE_DOCUMENTS_TABLE,
    CREATE_DOCUMENT_CHUNKS_TABLE,
    CREATE_CALL_SUMMARIES_TABLE,
    CREATE_ANALYTICS_TABLE,
    CREATE_AVAILABLE_SLOTS_TABLE,
    CREATE_INDEXES,
]
