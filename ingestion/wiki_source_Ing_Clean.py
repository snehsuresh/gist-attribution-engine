import requests
import hashlib
import pandas as pd
from datetime import datetime

topics = [
    "Russo-Ukrainian War", "China–Taiwan relations", "Gaza–Israel conflict", "North Korea", "NATO",
    "Trade war", "Refugee crisis", "Venezuelan crisis", "Myanmar coup", "Hong Kong protests",
    "Regulation of artificial intelligence", "Deepfake", "Facial recognition", "Social media censorship",
    "Data privacy", "Cybersecurity", "Cryptocurrency regulation", "Quantum computing", "Autonomous vehicles", "5G technology",
    "Climate change policy", "Renewable energy", "Carbon tax", "Deforestation", "Plastic pollution",
    "Geoengineering", "Wildfire", "Arctic ice melt", "Ocean acidification", "Endangered species",
    "COVID-19 vaccine", "Long COVID", "CRISPR gene editing", "Mental health", "mRNA technology",
    "Opioid epidemic", "Psychedelic therapy", "Vaccine hesitancy", "GMO crops", "Space exploration",
    "Interest rate", "Inflation", "Universal basic income", "Gig economy", "Remote work",
    "Minimum wage", "Stock market", "Housing crisis", "Global recession", "Automation",
    "Immigration policy", "Abortion rights", "Gun control", "Electoral fraud", "Supreme Court",
    "Privacy laws", "Term limits", "Voting rights", "Political polarization", "Public trust in government",
    "Black Lives Matter", "LGBTQ+ rights", "Me Too movement", "Social justice", "Cancel culture",
    "Nationalism", "Cultural appropriation", "Fake news", "Domestic terrorism", "Indigenous rights",
    "Student debt", "Online learning", "Charter schools", "Critical race theory", "Free speech on campus",
    "AI in education", "Standardized testing", "Teacher strikes", "Language preservation", "Aging population",
    "Human rights", "Privacy vs security", "Biofuels", "Electric vehicles", "Mental health stigma",
    "Income inequality", "National debt", "Blockchain", "Smart cities", "Internet regulation",
    "Globalization", "Supply chain", "Antitrust law", "Net neutrality", "Refugee policy",
    "Pandemic preparedness", "Food security", "Water scarcity", "Telemedicine", "Aging workforce",

    # From 'wikipedia_articles'
    "Cross-Strait relations", "Telecommuting", "Economy", "Central bank", "Monetary policy",
    "Fiscal policy", "Gross domestic product", "Economic recession", "Quantitative easing",
    "Unemployment", "Supply and demand", "Consumer price index", "Federal Reserve System",
    "Stagflation", "Keynesian economics", "Neoliberalism", "Economic stimulus", "Housing bubble",
    "Shadow banking system", "Venture capital", "Initial public offering", "Private equity",
    "E-commerce", "ESG investing", "Corporate governance", "Artificial intelligence", "Machine learning",
    "Deep learning", "Transformer (machine learning model)", "Natural language processing",
    "Large language model", "OpenAI", "GPT (language model)", "Neural network", "Algorithmic bias",
    "Technological unemployment", "Edge computing", "Internet of things", "Brain–computer interface", "Web3",
    "Surveillance capitalism", "General Data Protection Regulation", "Digital rights", "Algorithmic accountability",
    "Misinformation", "Internet censorship", "Digital divide", "Fairness (machine learning)", "Airline",
    "Low-cost carrier", "Airline deregulation", "Hub-and-spoke", "Airfare pricing", "Passenger load factor",
    "Transportation economics", "CARES Act", "Airport classification (FAA)", "Essential Air Service",
    "Journalism", "News aggregator", "Media bias", "Fact-checking", "Freedom of the press",
    "Echo chamber (media)", "Clickbait", "Media literacy", "Social media algorithm", "Big data",
    "Information overload", "Data visualization", "Statistical inference", "Exploratory data analysis",
    "Knowledge graph", "Information retrieval", "Semantic search", "Topic modeling", "Named entity recognition",
    "2008 financial crisis", "COVID-19 pandemic", "Great Depression", "Bretton Woods system", "Cold War",
    "United Nations", "G7", "World Bank", "International Monetary Fund", "Peer review",
    "Academic publishing", "Citation", "Scientific method", "Reproducibility", "Open access",
    "Literature review", "Meta-analysis", "Research ethics", "Stanford Encyclopedia of Philosophy"
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

        article_id = "wiki_" + hashlib.md5(f"Wikipedia_{published_at}_{title}".encode()).hexdigest()[:8]

        return {
            "id": article_id,
            "title": title,
            "source": "Wikipedia",
            "author": "Wikipedia editors",
            "published_at": published_at,
            "content": content,
            "section": "Encyclopedia",
            "tags": topic.lower().split()
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
