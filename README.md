# Sentiment Analysis on YouTube Comments using Machine Learning

## **Project Overview**
This project is a **YouTube comment sentiment analysis system** built using **Machine Learning** and **Flask**.  
It extracts comments from YouTube videos, classifies them as **Positive**, **Negative**, or **Neutral**, and displays the results on an interactive web dashboard.
The project also optionally generates **short AI summaries** of positive and negative comments using **Google Gemini API**.

## **Features**
- Extract YouTube video comments using YouTube Data API v3  
- Preprocess and clean comments for analysis  
- Classify comments into Positive, Negative, or Neutral using ML models  
- Calculate sentiment percentages and display them in a pie chart  
- Generate two-line summaries for positive and negative comments (via Google Gemini)  
- Interactive, responsive **Flask dashboard** with charts and summaries  

## **Folder Structure**

YouTube-Sentiment-Analysis-ML/
│
├─ app.py                 # Flask backend
├─ requirements.txt       # Python dependencies
├─ README.md              # Project documentation
├─ model.pkl              # Trained ML model
├─ vectorizer.pkl         # TF-IDF / CountVectorizer
├─ label_encoder.pkl      # Label encoder
│
├─ templates/
│   ├─ index.html         # Home page (input YouTube link)
│   └─ dash.html          # Dashboard page
│
├─ static/
│   ├─ dash.css           # Dashboard CSS
│   └─ images/            # Optional images/logo
│
└─ utils/                 # Optional helper scripts
    └─ youtube_api.py     # Fetch YouTube comments

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Add your **API keys** in `app.py`:
   - YouTube Data API key  
   - Google Gemini API key (optional)  
4. Run the Flask app:
```bash
python app.py
```
5. Open in browser: `http://127.0.0.1:5000`

---

## **Usage**
1. Enter a **YouTube video URL** in the home page input.  
2. Click **Analyze**.  
3. View **sentiment percentages**, **charts**, and **AI summaries** on the dashboard.  


## **License**
This project is licensed under the **MIT License**.  

---

## **Author**
**Barathkumar** – B.Tech AI & Data Science  

