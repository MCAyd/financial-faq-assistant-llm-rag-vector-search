import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

tz = ZoneInfo("Europe/Berlin")

def get_db_connection():
    try:
        # Try connecting with environment variables
        connection = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        print("Connection successful using environment variables")
    except Exception as e:
        print(f"Failed to connect using environment variables: {e}")
        print("Attempting to connect using localhost...")
        try:
            # If environment variable connection fails, connect using localhost
            connection = psycopg2.connect(
                host="localhost",
                database=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
            )
            print("Connection successful using localhost")
        except Exception as local_error:
            print(f"Failed to connect using localhost: {local_error}")
            connection = None

    return connection

def prep_db():
        try:
            conn = psycopg2.connect(
            host="localhost",
            database="postgres",
            port=os.getenv("POSTGRES_PORT"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"))
    
            db_name = os.getenv("POSTGRES_DB")
            conn.autocommit = True

            with conn.cursor() as cur:
                cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
                
                if cur.fetchone() is None:
                    cur.execute(f"CREATE DATABASE {db_name};")
                    print(f"Database created: {db_name}")
                else:
                    print(f"Database already exists: {db_name}")
        finally:
            conn.close()
            
                    
def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    company_id TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    openai_cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_conversation(conversation_id, question, answer_data, company_id, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations 
                (id, question, answer, company_id, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP))
            """,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    company_id,
                    answer_data["response_time"],
                    answer_data["relevance"],
                    answer_data["relevance_explanation"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    answer_data["eval_prompt_tokens"],
                    answer_data["eval_completion_tokens"],
                    answer_data["eval_total_tokens"],
                    answer_data["openai_cost"],
                    timestamp,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def save_feedback(conversation_id, feedback, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, COALESCE(%s, CURRENT_TIMESTAMP))",
                (conversation_id, feedback, timestamp),
            )
        conn.commit()
    finally:
        conn.close()


def get_recent_conversations(limit=5, relevance=None):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = """
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
            """
            if relevance:
                query += f" WHERE c.relevance = '{relevance}'"
            query += " ORDER BY c.timestamp DESC LIMIT %s"

            cur.execute(query, (limit,))
            return cur.fetchall()
    finally:
        conn.close()


def get_feedback_stats():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as thumbs_down
                FROM feedback
            """)
            return cur.fetchone()
    finally:
        conn.close()
