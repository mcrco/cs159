import random
import re
import os
import requests
from nltk.tokenize import sent_tokenize
from datasets import Dataset, DatasetDict
import nltk

# Ensure NLTK's sentence tokenizer is available
try:
    sent_tokenize("example sentence.")
except:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# ---- Step 1: Load and clean Bible text ----
# Define paths and URL
DATA_DIR = "texts/"
BOOK_FILENAME = "bible_king_james.txt"
BOOK_PATH = os.path.join(DATA_DIR, BOOK_FILENAME)
BOOK_URL = "https://www.gutenberg.org/cache/epub/10/pg10.txt"
OUTPUT_DATASET_DIR = "stego_dataset_bible/"

# Ensure the download directory exists
if not os.path.exists(DATA_DIR):
    print(f"Creating directory: {DATA_DIR}")
    os.makedirs(DATA_DIR)

# Download the book if it doesn't exist
if not os.path.exists(BOOK_PATH):
    print(f"Downloading The King James Bible from {BOOK_URL}...")
    try:
        response = requests.get(BOOK_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(BOOK_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Successfully downloaded and saved to {BOOK_PATH}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        exit(1)  # Exit if download fails
    except IOError as e:
        print(f"Error writing file: {e}")
        exit(1)  # Exit if writing fails
else:
    print(f"The King James Bible already downloaded at {BOOK_PATH}")

# Now proceed with reading the file
print(f"Loading text from {BOOK_PATH}...")
with open(BOOK_PATH, encoding="utf-8") as f:
    text = f.read()

# Remove Gutenberg header/footer
# Using regex from the provided websearch result for specific headers
text = re.split(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE \*\*\*", text, maxsplit=1, flags=re.IGNORECASE)[-1]
text = re.split(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE \*\*\*", text, maxsplit=1, flags=re.IGNORECASE)[0]


# Normalize spacing and remove excessive newlines
text = re.sub(r"\s+", " ", text).strip()

# Tokenize into sentences
sentences = sent_tokenize(text)

# Filter sentences (e.g., length between 5 and 30 words)
# This range might need adjustment for Bible text.
# Also, Bible verses often start with "chapter:verse " which we might want to remove or handle.
# For now, let's keep a basic filter.
processed_sentences = []
for s in sentences:
    # Remove verse numbers like "1:2 " or "10:25 " from the beginning of sentences
    s_cleaned = re.sub(r"^\d+:\d+\s+", "", s.strip())
    if 5 <= len(s_cleaned.split()) <= 30:
        processed_sentences.append(s_cleaned)

sentences = list(dict.fromkeys(processed_sentences))  # Deduplicate

print(f"Found {len(sentences)} suitable sentences after filtering and deduplication.")

# ---- Step 2: Generate examples with bit=0 and bit=1 ----
data = []
for sent in sentences:
    for bit in (0, 1):
        data.append({"buffer_text": sent, "bit": bit})

random.seed(42)
random.shuffle(data)

# ---- Step 3: Split and save as HuggingFace dataset ----
n = len(data)
if n == 0:
    print("No data to save. Exiting.")
    exit()

train_split = int(n * 0.8)
val_split = int(n * 0.9)

train_data = data[:train_split]
val_data = data[train_split:val_split]
test_data = data[val_split:]

print(f"Dataset split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

ds = DatasetDict(
    {
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    }
)

if not os.path.exists(OUTPUT_DATASET_DIR):
    os.makedirs(OUTPUT_DATASET_DIR)

ds.save_to_disk(OUTPUT_DATASET_DIR)
print(f"✅ Saved dataset with {len(data)} examples to {OUTPUT_DATASET_DIR}") 