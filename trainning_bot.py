import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the CSV file
df = pd.read_csv("/multilnag.csv", on_bad_lines='skip')

# Define the core languages
core_langs = ["English", "Hindi", "Marathi", "Tamil", "Japanese"]

for lang in core_langs:
    # Filter out rows where source language text is missing
    df_lang = df[~df[lang].isna()]
    sentences = df_lang[lang].astype(str).tolist()
    indices = df_lang.index.tolist()  # Keep track of original df rows

    print(f"Training vectorizer for {lang} on {len(sentences)} sentences...")

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Save vectorizer
    with open(f"vectorizer_{lang.lower()}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    # Save vectors
    with open(f"X_{lang.lower()}.pkl", "wb") as f:
        pickle.dump(X, f)
    # Save sentences
    with open(f"sentences_{lang.lower()}.pkl", "wb") as f:
        pickle.dump(sentences, f)
    # Save indices (to map back to original dataframe)
    with open(f"indices_{lang.lower()}.pkl", "wb") as f:
        pickle.dump(indices, f)

# Save the full dataframe for aligned lookup
df.to_pickle("full_translations.pkl")

print("✅ All vectorizers and data saved.")
