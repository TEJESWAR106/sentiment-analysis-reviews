import pandas as pd
import os

def load_data():
    csv_path = "tweets.csv"

    # If CSV already downloaded, use it
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[['text', 'airline_sentiment']].rename(
            columns={'airline_sentiment': 'sentiment'})
        df = df.dropna()
        return df

    # Try multiple URLs
    urls = [
        "https://raw.githubusercontent.com/bhuvanakundumani/twitter-airline-sentiment/master/data/Tweets.csv",
        "https://raw.githubusercontent.com/nicholasdavidson10/Twitter-Airline-Sentiment/master/Tweets.csv",
        "https://raw.githubusercontent.com/leanhdung1994/Deep-Learning/master/Tweets.csv"
    ]

    for url in urls:
        try:
            print(f"Trying: {url}")
            df = pd.read_csv(url)
            df.to_csv(csv_path, index=False)  # cache locally
            df = df[['text', 'airline_sentiment']].rename(
                columns={'airline_sentiment': 'sentiment'})
            df = df.dropna()
            print(f"Success! {len(df)} reviews loaded.")
            return df
        except Exception as e:
            print(f"Failed: {e}")
            continue

    # Fallback — generate sample data if all URLs fail
    print("All URLs failed. Using sample data...")
    import random
    random.seed(42)

    positive = [
        "Great flight experience!", "Loved the service", "Amazing staff very helpful",
        "Best airline ever", "Smooth journey excellent crew", "Very comfortable seats",
        "On time departure loved it", "Friendly staff great food",
        "Wonderful experience highly recommend", "Perfect flight no complaints",
        "Super fast boarding great service", "Excellent in flight entertainment",
        "Crew was amazing very attentive", "Great value for money",
        "Would definitely fly again", "Seats comfortable flight smooth",
        "Staff very professional and kind", "Quick boarding excellent service",
        "Flight was fantastic loved everything", "Best travel experience ever"
    ] * 100

    negative = [
        "Terrible experience never again", "Flight delayed no explanation",
        "Lost my baggage very upset", "Worst airline horrible service",
        "Staff rude and unhelpful", "Long wait no compensation",
        "Flight cancelled last minute", "Seats uncomfortable terrible food",
        "Customer service is awful", "Very disappointing experience",
        "Hours delayed no updates", "Baggage fees ridiculous",
        "Rude staff zero help", "Never flying this airline again",
        "Complete disaster avoid this airline", "Worst flight of my life",
        "No apology for massive delay", "Food was terrible overpriced",
        "Gate changed three times chaos", "Missed connection due to delay"
    ] * 100

    neutral = [
        "Flight was okay nothing special", "Average experience as expected",
        "Decent service nothing great", "Flight on time seats okay",
        "Normal flight nothing to complain", "Basic service standard airline",
        "Got there safely thats it", "Mediocre experience overall",
        "Nothing special but fine", "Standard airline nothing more",
        "Okay experience would consider again", "Flight fine service average",
        "Unremarkable but acceptable trip", "Got to destination okay",
        "Normal boarding average seats", "Middle of the road airline",
        "Fine for the price paid", "Nothing outstanding nothing terrible",
        "Acceptable flight standard service", "Typical airline experience"
    ] * 100

    texts = positive + negative + neutral
    sentiments = (["positive"] * len(positive) +
                  ["negative"] * len(negative) +
                  ["neutral"] * len(neutral))

    df = pd.DataFrame({"text": texts, "sentiment": sentiments})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Sample dataset created: {len(df)} reviews")
    return df