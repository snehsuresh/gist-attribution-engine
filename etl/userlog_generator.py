#!/usr/bin/env python3
"""
simulate_user_logs.py: Generate realistic simulated user interaction logs
for our Gist Attribution pipeline using DuckDB data.

Features:
- Random users and sessions
- Realistic queries based on article tags
- Top-K retrieval from articles table
- Simulated views, clicks, chunk-level views, and feedback events
- Timestamp generation aligned with session flow and article publish dates
- Insertion into DuckDB `user_events` table for downstream attribution testing
"""

import duckdb
import random
import datetime
from uuid import uuid4

# Configuration
DB_PATH = "data/processed/duckdb/articles.duckdb"
NUM_USERS = 20                       # number of simulated users
QUERIES_PER_SESSION = (1, 4)        # min and max queries per session
TOP_K = 10                          # Top-K articles to retrieve per query
MIN_VIEWS = 3                       # min number of viewed results
MAX_VIEWS = 6                       # max number of viewed results
MAX_CLICKS = 2                      # max number of clicked results among views
SESSION_TIMEOUT_MINUTES = 30        # session window size
FEEDBACK_PROB = 0.3                 # probability of feedback event after a click

# Query templates for natural phrasing
QUERY_TEMPLATES = [
    "What's new in {topic} right now?",
    "Latest updates on {topic}",
    "Tell me about {topic}",
    "How is {topic} evolving?",
    "What's happening with {topic}?"
]

# Connect to DuckDB
conn = duckdb.connect(DB_PATH)

# Prepare user_events table
conn.execute("DROP TABLE IF EXISTS user_events;")
conn.execute("""
CREATE TABLE user_events (
    user_id TEXT,
    session_id TEXT,
    query TEXT,
    timestamp TIMESTAMP,
    article_id TEXT,
    chunk_id TEXT,
    event_type TEXT
);
""")

# Fetch distinct individual tags
rows = conn.execute("SELECT DISTINCT tags FROM articles;").fetchall()
all_tags = set()
for (tag_str,) in rows:
    for t in tag_str.split(','):
        all_tags.add(t.strip())
all_tags = [t for t in all_tags if t]

# Function to sample a realistic timestamp within session scope
def sample_timestamp(base_time, delta_min=10, delta_max=300):
    return base_time + datetime.timedelta(seconds=random.randint(delta_min, delta_max))

events = []

# Simulate interactions per user
for _ in range(NUM_USERS):
    user_id = f"user_{uuid4().hex[:8]}"
    # Each user has 1-2 sessions
    for _ in range(random.randint(1, 2)):
        session_id = f"sess_{uuid4().hex[:8]}"
        # Random session start between article date range
        session_start = datetime.datetime(2025, 6, 1) + datetime.timedelta(
            days=random.randint(0, 49),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        num_queries = random.randint(*QUERIES_PER_SESSION)
        for q_idx in range(num_queries):
            # Choose a topic and form a query
            topic = random.choice(all_tags)
            query = random.choice(QUERY_TEMPLATES).format(topic=topic)
            # Retrieve Top-K relevant articles by tag
            sql = f"""
                SELECT id, published_at
                FROM articles
                WHERE tags LIKE '%{topic}%'
                ORDER BY published_at DESC
                LIMIT {TOP_K};
            """
            results = conn.execute(sql).fetchall()
            article_ids = [r[0] for r in results]
            if not article_ids:
                continue

            # Simulate views and clicks
            num_views = random.randint(MIN_VIEWS, min(MAX_VIEWS, len(article_ids)))
            viewed = random.sample(article_ids, num_views)
            num_clicks = random.randint(1, min(MAX_CLICKS, len(viewed)))
            clicked = random.sample(viewed, num_clicks)

            # Define a sub-session time slot for this query
            query_base = session_start + datetime.timedelta(
                minutes=q_idx * (SESSION_TIMEOUT_MINUTES / max(1, num_queries))
            )

            # Record view/click events
            for aid in viewed:
                event_type = 'click' if aid in clicked else 'view'
                evt_time = sample_timestamp(query_base)
                events.append((user_id, session_id, query, evt_time, aid, None, event_type))

                # For clicked articles, simulate chunk-level views
                if event_type == 'click':
                    chunks = conn.execute(
                        f"SELECT chunk_id FROM article_chunks WHERE article_id = '{aid}' ORDER BY chunk_index LIMIT 5;"
                    ).fetchall()
                    chunk_ids = [c[0] for c in chunks]
                    sampled_chunks = random.sample(
                        chunk_ids, min(len(chunk_ids), random.randint(1, 3))
                    )
                    for cid in sampled_chunks:
                        c_time = sample_timestamp(evt_time, 5, 60)
                        events.append((user_id, session_id, query, c_time, aid, cid, 'chunk_view'))

            # Simulate feedback events for clicked articles
            for aid in clicked:
                if random.random() < FEEDBACK_PROB:
                    fb_time = sample_timestamp(query_base, 300, 600)
                    fb_type = random.choice(['feedback_useful', 'feedback_confusing'])
                    events.append((user_id, session_id, query, fb_time, aid, None, fb_type))

# Bulk insert events into user_events
conn.executemany(
    "INSERT INTO user_events VALUES (?, ?, ?, ?, ?, ?, ?);",
    events
)
conn.close()
print(f"✅ Inserted {len(events)} simulated user events into DuckDB.")
