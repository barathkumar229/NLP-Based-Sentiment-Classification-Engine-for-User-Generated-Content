from flask import Flask, render_template, request
from urllib.parse import urlparse, parse_qs
import requests
import pickle
import google.generativeai as genai
import json
import re
import os
import tempfile

app = Flask(__name__)

# Disable caching so BACK BUTTON does not reload dashboard
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Load ML model, vectorizer, encoder
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract YouTube video ID
def get_video_id(url):
    if not url:
        return None
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return None

# HOME PAGE
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

# DASHBOARD PAGE
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    link = request.form.get("youtube_link") or request.args.get("link")
    total = positive_pct = negative_pct = neutral_pct = 0
    positive_summary = ""
    negative_summary = ""
    overall_summary = ""

    video_id = get_video_id(link)
    if video_id:
        # Load YouTube API key from environment
        API_KEY = os.environ.get("YOUTUBE_API_KEY")
        if not API_KEY:
            return "YouTube API key not set in environment variables.", 500

        # Fetch comments
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100"
        response = requests.get(url)
        data = response.json()

        comments = []
        for item in data.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            comments.append(comment.strip())

        neutral_comments = []
        positive_comments = []
        negative_comments = []

        # Predict sentiment
        for comment in comments:
            new_vec = vectorizer.transform([comment])
            y_pred_new = model.predict(new_vec)
            original_label = label_encoder.inverse_transform(y_pred_new)[0]
            if original_label == "Positive":
                positive_comments.append(comment)
            elif original_label == "Negative":
                negative_comments.append(comment)
            else:
                neutral_comments.append(comment)

        total = len(comments)
        if total > 0:
            positive_pct = len(positive_comments) / total * 100
            negative_pct = len(negative_comments) / total * 100
            neutral_pct = len(neutral_comments) / total * 100

        # Configure Gemini using service account JSON
        gemini_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if gemini_json:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
                f.write(gemini_json.encode())
                temp_path = f.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path

        genai.configure()  # Automatically uses service account JSON
        gemini = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = f"""
        Analyze the comments and return ONLY a JSON object with this structure:
        {{
          "positive": "LINE 1\\nLINE 2",
          "negative": "LINE 1\\nLINE 2",
          "summary": "LINE 1\\nLINE 2"
        }}
        STRICT RULES:
        - Each field must be a single STRING with EXACTLY TWO LINES.
        - DO NOT return a JSON array or brackets [].
        - Do NOT add bullet points or emojis.
        - No extra text outside JSON.

        Positive comments:
        {chr(10).join(positive_comments)}

        Negative comments:
        {chr(10).join(negative_comments)}
        """

        response_text = gemini.generate_content(prompt).text.strip()

        try:
            json_str = re.search(r"\{.*\}", response_text, re.DOTALL).group(0)
            data = json.loads(json_str)
            positive_summary = data.get("positive", "")
            negative_summary = data.get("negative", "")
            overall_summary = data.get("summary", "")
        except Exception as e:
            print("JSON PARSE ERROR:", e)
            print("MODEL OUTPUT:", response_text)
            positive_summary = "No positive summary available."
            negative_summary = "No negative summary available."
            overall_summary = "No overall summary available."

    return render_template(
        "dash.html",
        link=link,
        total=total,
        positive_pct=positive_pct,
        negative_pct=negative_pct,
        neutral_pct=neutral_pct,
        positive_summary=positive_summary,
        negative_summary=negative_summary,
        overall_summary=overall_summary
    )

if __name__ == "__main__":
    app.run(debug=True)
