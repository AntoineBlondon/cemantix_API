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
CORS(app)


# Hugging Face model link
MODEL_URL = "https://huggingface.co/AzureBlondon/cemantix-model/resolve/main/model.bin"

def download_model():
    """Downloads the model from Hugging Face and loads it from a temporary file."""
    print(f"Downloading model from {MODEL_URL}...")

    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 KB
    downloaded_size = 0

    # Create a temporary file for the model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as temp_file:
        temp_path = temp_file.name  # Save path to load later
        
        start = time.time()
        for chunk in response.iter_content(block_size):
            if chunk:
                temp_file.write(chunk)
                downloaded_size += len(chunk)
                percent_complete = (downloaded_size / total_size) * 100 if total_size else 0
                print(f"\rDownloading: {percent_complete:.2f}% [{downloaded_size / (1024 * 1024):.2f}MB]", end="")

        print("\nDownload complete. Loading model from file...")

    # Load the model from the saved file
    model = KeyedVectors.load_word2vec_format(temp_path, binary=True, unicode_errors="ignore")

    # Remove the temporary file to free up space
    os.remove(temp_path)

    end = time.time()
    print(f"Model loaded in {end - start:.2f} seconds")
    return model

# Load the model on startup
model = download_model()


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
    return variants[0] if variants else word





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
    guess = data.get("guess")
    OG_guess = guess
    
    if not guess:
        return jsonify({"error": "Missing guess"}), 400
    
    target = get_most_frequent_variant(get_daily_random_word(model), model)
    guess = get_most_frequent_variant(guess, model) 
    
    if target not in model or guess not in model:
        return jsonify({"error": "One or both words are not in the model's vocabulary"}), 400
    
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
