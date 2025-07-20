import requests
import pandas as pd
from datetime import datetime
import time

# --- Config ---
API_KEY = ""
BASE_URL = "http://api.mediastack.com/v1/news"

# --- Topics to search ---
topics = [
    "Russo-Ukrainian War",
    "China–Taiwan relations",
    "Gaza–Israel conflict",
    "North Korea",
    "NATO",
    "Trade war",
    "Refugee crisis",
    "Venezuelan crisis",
    "Myanmar coup",
    "Hong Kong protests",
    "Regulation of artificial intelligence",
    "Facial recognition",
    "Social media censorship",
    "Data privacy",
    "Cybersecurity",
    "Cryptocurrency regulation",
    "Quantum computing",
    "Autonomous vehicles",
    "5G technology",
    "Renewable energy",
    "Carbon tax",
    "Deforestation",
    "Plastic pollution",
    "Geoengineering",
    "Wildfire",
    "Arctic ice melt",
    "Ocean acidification",
    "Endangered species",
    "Long COVID",
    "CRISPR gene editing",
    "Mental health",
    "mRNA technology",
    "Opioid epidemic",
    "Psychedelic therapy",
    "Vaccine hesitancy",
    "GMO crops",
    "Space exploration",
    "Universal basic income",
    "Gig economy",
    "Minimum wage",
    "Stock market",
    "Housing crisis",
    "Global recession",
    "Automation",
    "Gun control",
    "Electoral fraud",
    "Supreme Court",
    "Privacy laws",
    "Voting rights",
    "Political polarization",
    "Black Lives Matter",
    "LGBTQ+ rights",
    "Me Too movement",
    "Cancel culture",
    "Fake news",
    "Domestic terrorism",
    "Indigenous rights",
    "Student debt",
    "Online learning",
    "Critical race theory",
    "AI in education",
    "Aging population",
    "Electric vehicles",
    "Income inequality",
    "National debt",
    "Blockchain",
    "Smart cities",
    "Internet regulation",
    "Globalization",
    "Supply chain",
    "Antitrust law",
    "Net neutrality",
    "Refugee policy",
    "Pandemic preparedness",
    "Food security",
    "Water scarcity",
    "Telemedicine",
]

# --- Container for all news ---
all_articles = []

# --- Loop through each topic ---
for topic in topics:
    print(f"Fetching news for topic: {topic}")

    params = {
        "access_key": API_KEY,
        "keywords": topic,
        "countries": "us",
        "languages": "en",
        "limit": 50,  # To avoid overload; can increase
        "sort": "published_desc",
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch for: {topic}")
        continue

    data = response.json()

    for i, article in enumerate(data.get("data", [])):
        all_articles.append(
            {
                "id": f"{article.get('source', 'unknown')}_{article.get('published_at', '')[:10]}_{i}",
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "author": article.get("author") or "Unknown",
                "published_at": article.get("published_at", "")[:10],
                "content": article.get("description", "") or "<no content>",
                "section": article.get("category", "General"),
                "tags": [topic.lower()],  # Add topic as tag
            }
        )

    # Respect API rate limit
    time.sleep(1)  # Sleep to avoid hitting rate limits

# --- Save to CSV ---
df = pd.DataFrame(all_articles)
df.to_csv("mediastack_topics_news.csv", index=False)
print(f"Saved {len(df)} articles to 'mediastack_topics_news.csv'")
