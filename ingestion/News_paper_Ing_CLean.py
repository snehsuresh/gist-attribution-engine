import requests
import hashlib
import pandas as pd

API_KEY = "e6bf9deb77844f94b982e4882903c722"  # Replace with your key
START_DATE = "2025-06-20"
SOURCE_PREFIX = "newsapi"

TOPICS = [
    "Ukraine war", "AI regulation", "climate policy", "deepfakes",
    "interest rates", "immigration policy", "COVID vaccine",
    "abortion rights", "China Taiwan", "remote work", "economy", "inflation",
    "Russo-Ukrainian War", "China–Taiwan relations", "Gaza–Israel conflict",
    "North Korea", "NATO", "Trade war", "Refugee crisis", "Venezuelan crisis",
    "Myanmar coup", "Hong Kong protests", "Regulation of artificial intelligence",
    "Facial recognition", "Social media censorship", "Data privacy", "Cybersecurity",
    "Cryptocurrency regulation", "Quantum computing", "Autonomous vehicles",
    "5G technology", "Renewable energy", "Carbon tax", "Deforestation",
    "Plastic pollution", "Geoengineering", "Wildfire", "Arctic ice melt",
    "Ocean acidification", "Endangered species", "Long COVID", "CRISPR gene editing",
    "Mental health", "mRNA technology", "Opioid epidemic", "Psychedelic therapy",
    "Vaccine hesitancy", "GMO crops", "Space exploration", "Universal basic income",
    "Gig economy", "Minimum wage", "Stock market", "Housing crisis",
    "Global recession", "Automation", "Gun control", "Electoral fraud", "Supreme Court",
    "Privacy laws", "Voting rights", "Political polarization", "Black Lives Matter",
    "LGBTQ+ rights", "Me Too movement", "Cancel culture", "Fake news",
    "Domestic terrorism", "Indigenous rights", "Student debt", "Online learning",
    "Critical race theory", "AI in education", "Aging population", "Electric vehicles",
    "Income inequality", "National debt", "Blockchain", "Smart cities", "Internet regulation",
    "Globalization", "Supply chain", "Antitrust law", "Net neutrality", "Refugee policy",
    "Pandemic preparedness", "Food security", "Water scarcity", "Telemedicine"
]

def generate_id(source, published_at, title):
    raw = f"{source}_{published_at}_{title}"
    return SOURCE_PREFIX + "_" + hashlib.md5(raw.encode()).hexdigest()[:8]

def fetch_articles_for_topic(topic):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "from": START_DATE,
        "sortBy": "popularity",
        "pageSize": 10,
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        print(f"Error fetching for topic {topic}: {response.status_code}")
        return []

def process_articles(articles, topic):
    data = []
    for article in articles:
        if not article.get("content"):
            continue
        article_id = generate_id(article["source"]["name"], article["publishedAt"], article["title"])
        data.append({
            "id": article_id,
            "title": article["title"],
            "source": article["source"]["name"],
            "author": article.get("author"),
            "published_at": article["publishedAt"][:10],
            "content": article["content"],
            "section": topic,
            "tags": [topic.lower()]
        })
    return data

# Main loop
all_articles = []
for topic in TOPICS:
    raw_articles = fetch_articles_for_topic(topic)
    processed = process_articles(raw_articles, topic)
    all_articles.extend(processed)

# Convert to DataFrame
df = pd.DataFrame(all_articles)

# Show preview
print(df.head(3))
df.to_csv("news_api_final.csv", index=False)