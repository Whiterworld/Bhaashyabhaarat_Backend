from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os
import pickle
import pandas as pd
import jiwer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

app = Flask(__name__)
#CORS(app)  # In production, configure CORS properly
CORS(app, resources={r"/*": {"origins": ["https://bhaashyabhaarat.netlify.app"]}})


# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client.Whiter
users_collection = db.users

# Supported languages
core_langs = ["English", "Hindi", "Marathi", "Tamil", "Japanese"]

# Load translation data
df = pd.read_pickle("all_pickles/full_translations.pkl")

vectorizers = {}
vectors = {}
sentences = {}
indices = {}

for lang in core_langs:
    lang_key = lang.lower()
    with open(f"all_pickles/vectorizer_{lang_key}.pkl", "rb") as f:
        vectorizers[lang] = pickle.load(f)
    with open(f"all_pickles/X_{lang_key}.pkl", "rb") as f:
        vectors[lang] = pickle.load(f)
    with open(f"all_pickles/sentences_{lang_key}.pkl", "rb") as f:
        sentences[lang] = pickle.load(f)
    with open(f"all_pickles/indices_{lang_key}.pkl", "rb") as f:
        indices[lang] = pickle.load(f)

target_mapping = {
    "English": ["Hindi", "Marathi", "Tamil", "Japanese"],
    "Hindi": ["English", "Marathi", "Tamil", "Japanese"],
    "Marathi": ["English", "Hindi", "Tamil", "Japanese"],
    "Tamil": ["English", "Hindi", "Marathi", "Japanese"],
    "Japanese": ["English", "Hindi", "Marathi", "Tamil"]
}

roman_map = {
    "Hindi": "HinEnglish",
    "Marathi": "MarathiEnglish",
    "Tamil": "TamilEnglish",
    "Japanese": "Romanization"
}

def normalize_language(lang):
    return lang.strip().capitalize()


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "Translation API is running"}), 200


@app.route("/languages", methods=["GET"])
def get_languages():
    return jsonify({
        "source_languages": core_langs,
        "target_mapping": target_mapping
    })


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    source_input = normalize_language(data.get("source_lang", ""))
    target_input = normalize_language(data.get("target_lang", ""))
    user_input = data.get("sentence", "").strip()

    if source_input not in core_langs or target_input not in core_langs:
        return jsonify({"error": "Invalid source or target language"}), 400

    if target_input not in target_mapping[source_input]:
        return jsonify({"error": f"Target language '{target_input}' not allowed for source '{source_input}'"}), 400

    if not user_input:
        return jsonify({"error": "Empty input sentence"}), 400

    vectorizer = vectorizers[source_input]
    X_source = vectors[source_input]
    source_sentences = sentences[source_input]
    source_indices = indices[source_input]

    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, X_source)
    best_idx = similarities.argmax()
    best_score = similarities[0, best_idx]
    best_sentence = source_sentences[best_idx]
    df_row_index = source_indices[best_idx]

    native_translation = df.loc[df_row_index, target_input]
    romanized_translation = df.loc[df_row_index, roman_map.get(target_input)] if target_input in roman_map else None

    if romanized_translation:
        combined_translation = f"{native_translation}\n({romanized_translation})"
    else:
        combined_translation = native_translation

    return jsonify({
        "input": user_input,
        "matched_source": best_sentence,
        "similarity": round(float(best_score), 3),
        "translation": combined_translation,
        "source_lang": source_input,
        "target_lang": target_input
    })


@app.route("/score-pronunciation", methods=["POST"])
def score_pronunciation():
    data = request.get_json()
    spoken = data.get("spoken", "").strip().lower()
    reference = data.get("reference", "").strip().lower()

    if not spoken or not reference:
        return jsonify({"error": "Missing spoken or reference text"}), 400

    wer = jiwer.wer(reference, spoken)
    score = max(0, 100 - wer * 100)

    comment = (
        "Excellent!" if score > 90 else
        "Good effort!" if score > 70 else
        "Try again more clearly."
    )

    return jsonify({
        "spoken": spoken,
        "reference": reference,
        "score": round(score, 2),
        "feedback": comment
    })


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already exists"}), 400

    hashed_password = generate_password_hash(password)
    user = {"name": name, "email": email, "password": hashed_password}
    users_collection.insert_one(user)

    return jsonify({"message": "User created successfully"}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"error": "Invalid credentials"}), 401

    user_data = {
        "id": str(user['_id']),
        "name": user['name'],
        "email": user['email']
    }

    return jsonify({"message": "Login successful", "user": user_data}), 200


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    load_dotenv()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))




# Without resources={r"/*": {"origins": [...]}}, your API is open to requests from any domain, which can be a security risk in production.
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": ["https://your-netlify-app.netlify.app"]}})