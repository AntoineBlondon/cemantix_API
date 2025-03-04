from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from gensim.models import KeyedVectors
import random
import datetime
import requests
import tempfile
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    """Ensure all responses include proper CORS headers."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Explicitly handle OPTIONS preflight requests to avoid 500 errors
@app.route("/guess", methods=["OPTIONS"])
@app.route("/daily-random-word", methods=["OPTIONS"])
def handle_options():
    """Returns a successful CORS preflight response."""
    response = jsonify({"message": "CORS preflight successful."})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response, 200


# Load the model
print("Loading small model...")
start = time.time()
model = KeyedVectors.load_word2vec_format("small_model.bin", binary=True)
end = time.time()
print(f"✅ Loaded small model in {end - start:.2f} seconds")

# Helper functions
def get_random_frequent_word(word_vectors, top_n=3000):
    """Return a random word from the top N most frequent words in the model."""
    if top_n > len(word_vectors.index_to_key):
        top_n = len(word_vectors.index_to_key)
    return random.choice(word_vectors.index_to_key[:top_n]).split("_")[0]

def get_daily_random_word(word_vectors, top_n=3000):
    """Return a random word that remains the same for the day."""
    random.seed(datetime.date.today().toordinal())
    return get_random_frequent_word(word_vectors, top_n)

def count_words_closer(target, guess, word_vectors):
    """Count how many words are closer to the target than the guessed word."""
    if target not in word_vectors or guess not in word_vectors:
        return {"error": "One of the words is not in the vocabulary!"}
    
    guess_distance = word_vectors.distance(target, guess)
    closer_count = sum(1 for word in word_vectors.index_to_key 
                       if word != target and word_vectors.distance(target, word) < guess_distance)
    return closer_count

def get_most_frequent_variant(word, word_vectors):
    """Return the most frequent variant of a word (first match found in model's vocabulary)."""
    variants = [w for w in word_vectors.index_to_key if w.startswith(word + "_")]
    return treat_word(variants[0] if variants else word)


def treat_word(word: str) -> str:
    word = word.lower()
    word = word.strip()
    return word





###############################################
########### Flask API Endpoints ###############
###############################################

@app.route("/daily-random-word", methods=["GET", "POST"])
def daily_random_word():
    word = get_daily_random_word(model)
    return jsonify({"daily_random_word": word})

@app.route("/guess", methods=["POST"])
def compare_words():
    data = request.json
    guess = treat_word(data.get("guess"))
    OG_guess = guess
    
    if not guess:
        return jsonify({"error": "Missing guess"}), 400
    
    target = get_most_frequent_variant(get_daily_random_word(model), model)
    guess = get_most_frequent_variant(guess, model) 
    
    if guess not in model:
        return jsonify({"error": f"Guess is not in the model's vocabulary: {guess}"}), 400
    
    distance = model.distance(target, guess)
    nb_words_closer = count_words_closer(target, guess, model)
    nb_words_total = len(model.index_to_key)
    percentage = nb_words_closer / nb_words_total * 100
    
    return jsonify({
        "guess": OG_guess,
        "distance": distance,
        "words_closer_count": nb_words_closer,
        "percentage_closer": f"{percentage:.2f}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
