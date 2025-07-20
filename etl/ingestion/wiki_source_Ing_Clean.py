import requests
import hashlib
import pandas as pd
from datetime import datetime

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


def fetch_wikipedia_article(topic):
    title_slug = topic.replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_slug}"

    try:
        res = requests.get(url)
        if res.status_code == 404:
            print(f"[Page Not Found] {title_slug}")
            return None
        res.raise_for_status()
        data = res.json()

        title = data.get("title", topic)
        content = data.get("extract", "")
        published_at = data.get("timestamp", datetime.now().isoformat())[:10]

        article_id = (
            "wiki_"
            + hashlib.md5(f"Wikipedia_{published_at}_{title}".encode()).hexdigest()[:8]
        )

        return {
            "id": article_id,
            "title": title,
            "source": "Wikipedia",
            "author": "Wikipedia editors",
            "published_at": published_at,
            "content": content,
            "section": "Encyclopedia",
            "tags": topic.lower().split(),
        }

    except Exception as e:
        print(f"[Error] {topic}: {e}")
        return None


articles = []

for topic in topics:
    article = fetch_wikipedia_article(topic)
    if article:
        articles.append(article)
        print(f"[Success] {article['title']}")

df = pd.DataFrame(articles)
df.to_csv("wikipedia_articles.csv", index=False)
print(df[["title", "published_at"]])
