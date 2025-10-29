# 🛒 Amazon Fake Review Detection using BERT and XLNet

This project detects **fake and genuine reviews** from **Amazon product pages** using **BERT** and **XLNet** transformer models.
The system takes an **Amazon product link** as input, scrapes all the visible reviews and then classifies each as **Real** or **Fake** using pre-trained NLP models.
It finally displays the analysis results and a **visual accuracy comparison** on a simple **Flask web interface**.

---

## 🚀 Features

* 🔗 Accepts **Amazon product URL** directly from the user
* 🤖 Dual model classification using **BERT** and **XLNet**
* 📊 Visual **pie chart** showing real vs fake reviews
* 🧾 Displays all reviews categorized as **Real** and **Fake**
* 💾 Automatically saves reviews to `amazon_reviews.csv`
* 🌐 Simple Flask-based web interface for easy interaction

---

## 🧠 Tech Stack

| Component           | Technology                                   |
| ------------------- | -------------------------------------------- |
| **Backend**         | Python (Flask)                               |
| **Models**          | BERT, XLNet (`transformers` by Hugging Face) |
| **Web Scraping**    | BeautifulSoup4, Requests                     |
| **Data Processing** | pandas, NumPy                                |
| **Visualization**   | Matplotlib                                   |
| **Frontend**        | HTML, CSS, Bootstrap                         |

---

## ⚙️ Installation and Setup

### 1. Clone this repository

### 2. Open the project folder

### 3. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
# or
source venv/bin/activate  # For Mac/Linux
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

### 6. Open in browser

Visit:

```
http://127.0.0.1:5000
```

---

## 🧩 How It Works

1. Enter a valid **Amazon product link** (e.g. `https://www.amazon.in/...`)
2. The scraper extracts visible **customer reviews** from the page.
3. Reviews are passed through **BERT** and **XLNet** models for classification.
4. Results are displayed:

   * **Real Reviews**
   * **Fake Reviews**
   * **Pie Chart Visualization**
   * **Status & Grade**

---

## 📊 Future Enhancements

* Add **RoBERTa** model for better comparison.
* Improve scraping for products with pagination.
* Support for **Flipkart**, **Myntra**, and other sites.
