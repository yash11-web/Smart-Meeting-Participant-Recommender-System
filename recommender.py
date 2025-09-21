import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def nltk_preprocess(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation, then lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return ' '.join(cleaned_tokens)

# Load extended participant profiles
df = pd.read_csv("participants.csv")

df['profile'] = df['skills'] + "; " + df['role'] + "; " + df['department'] + "; " + df['seniority']


df['processed_profile'] = df['profile'].apply(nltk_preprocess)

# Fit TF-IDF on preprocessed profiles
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500)
profile_mat = vectorizer.fit_transform(df['processed_profile'])

def recommend(meeting_desc, top_n=5):
    
    processed_desc = nltk_preprocess(meeting_desc)
    
    desc_vec = vectorizer.transform([processed_desc])
    
    sims = cosine_similarity(desc_vec, profile_mat).flatten()
    
    idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for i in idx:
        sim = sims[i]
        
        prof_terms = vectorizer.transform([df.loc[i,'processed_profile']]).toarray()[0]
        desc_terms = desc_vec.toarray()[0]
        contrib = prof_terms * desc_terms
        top_terms = [vectorizer.get_feature_names_out()[j] for j in contrib.argsort()[::-1] if contrib[j] > 0][:3]
        reason = f"These {', '.join(top_terms)} Skills are matched."
        results.append({
            'name': df.loc[i,'name'],
            'role': df.loc[i,'role'],
            'department': df.loc[i,'department'],
            'similarity': float(sim),
            'reason': reason
        })
    return results


if __name__ == "__main__":
    example_meeting = "Optimize cloud deployment for microservices and implement CI/CD pipelines."
    recs = recommend(example_meeting)
    print(f"Meeting description:\n{example_meeting}\n")
    print("Top recommendations:")
    for r in recs:
        print(f"{r['name']} ({r['role']}, {r['department']}) - Similarity: {r['similarity']:.3f}")
        print(f"Reason: {r['reason']}\n")
