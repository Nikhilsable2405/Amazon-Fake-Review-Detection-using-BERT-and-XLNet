import logging
import os
import torch
from flask import Flask, render_template, request
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import time
import random

# -------------------- Setup --------------------
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

global grade_color

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
]

headers = {
    "User-Agent": random.choice(USER_AGENTS),
    "Accept-Language": "en-US,en;q=0.9",
}

# -------------------- Review Extraction --------------------
def get_amazon_reviews(url, max_pages=1):
    reviews = []
    product_name = "Unknown Product"

    try:
        # Use your working headers setup
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # ✅ Product name
            product_title_element = soup.find("span", {"id": "productTitle"})
            if product_title_element:
                product_name = product_title_element.text.strip()

            # ✅ Reviews
            review_elements = soup.select('span[data-hook="review-body"]')
            print(f"📝 Total reviews found: {len(review_elements)}")

            if not review_elements:
                logging.warning("⚠ No reviews found — possible structure change.")
            else:
                for box in review_elements:
                    text = box.get_text(strip=True)
                    if text:
                        reviews.append({'review': text})

        else:
            logging.warning(f"⚠ Amazon blocked the request (Status: {response.status_code})")

    except Exception as e:
        logging.error(f"⚠ Error fetching reviews: {e}")

    return reviews, product_name


# -------------------- Save Reviews --------------------
def save_reviews_to_csv(reviews, filename):
    df = pd.DataFrame(reviews, columns=["review"])
    df.to_csv(filename, index=False)
    logging.info(f"✅ Saved {len(reviews)} reviews to {filename}")

# -------------------- Load Model --------------------
def load_model(model_path):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info(f"Model and tokenizer loaded from {model_path}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise ValueError(f"Error loading model from {model_path}: {e}")

# -------------------- Review Processing --------------------
def process_reviews(file_path, model, tokenizer, model_name):
    df = pd.read_csv(file_path)
    if 'review' not in df.columns:
        raise ValueError("CSV must contain 'review' column")

    batch_size = 16
    predictions = []
    reviews = df['review'].tolist()

    for i in range(0, len(reviews), batch_size):
        batch = [str(r).strip() for r in reviews[i:i + batch_size] if str(r).strip()]
        if not batch:
            continue

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).tolist()
        predictions.extend(preds)

    df['prediction'] = predictions
    processed_filename = f'{model_name}_processed_reviews.csv'
    df.to_csv(processed_filename, index=False)

    real_count = predictions.count(0)
    fake_count = predictions.count(1)
    grade, grade_color = calculate_grade(real_count, fake_count)
    logging.info(f"✅ Real: {real_count}, Fake: {fake_count}")

    return real_count, fake_count, grade, grade_color

# -------------------- Grading System --------------------
def calculate_grade(real_count, fake_count):
    total = real_count + fake_count
    if total == 0:
        return "No reviews", "gray"
    pct = (real_count / total) * 100
    if pct >= 80:
        return "A", "green"
    elif pct >= 60:
        return "B", "blue"
    elif pct >= 40:
        return "C", "orange"
    elif pct >= 20:
        return "D", "red"
    else:
        return "F", "darkred"

# -------------------- Flask Routes --------------------
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/BERT.html', methods=['GET', 'POST'])
def bert_view():
    url = request.form.get('url') or request.args.get('url')
    if not url:
        return render_template("BERT.html", error="No URL provided")

    reviews, product_name = get_amazon_reviews(url)
    if not reviews:
        return render_template("BERT.html", error="No reviews found or Amazon blocked the request.")

    save_reviews_to_csv(reviews, "amazon_reviews.csv")

    model_name = "bert-base-uncased"
    model_path = "model_checkpoints/Bert"
    model, tokenizer = load_model(model_path)

    real_count, fake_count, grade, grade_color = process_reviews("amazon_reviews.csv", model, tokenizer, model_name)

    results = {
        "status_message": "Processing completed.",
        "grade": grade
    }

    return render_template(
        'BERT.html',
        results=results,
        product_name=product_name,
        real_count=real_count,
        fake_count=fake_count,
        url=url
    )


@app.route('/XLnet.html', methods=['GET', 'POST'])
def XLnet_view():
    url = request.form.get('url') or request.args.get('url')
    if not url:
        return render_template("XLnet.html", error="No URL provided")

    reviews, product_name = get_amazon_reviews(url)
    if not reviews:
        return render_template("XLnet.html", error="No reviews found or Amazon blocked the request.")

    save_reviews_to_csv(reviews, "amazon_reviews.csv")

    model_name = "xlnet-base-cased"
    model_path = "model_checkpoints/XLnet"
    model, tokenizer = load_model(model_path)

    real_count, fake_count, grade, grade_color = process_reviews("amazon_reviews.csv", model, tokenizer, model_name)

    results = {
        "status_message": "Processing completed.",
        "grade": grade
    }

    return render_template(
        'XLnet.html',
        results=results,
        product_name=product_name,
        real_count=real_count,
        fake_count=fake_count,
        url=url
    )


# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run(debug=True)
