import random
import re
import os
import requests
from nltk.tokenize import sent_tokenize
from datasets import Dataset, DatasetDict

# ---- Step 1: Load and clean Wizard of Oz text ----
# Define paths and URL
DATA_DIR = "texts/"
BOOK_PATH = os.path.join(DATA_DIR, "wizard_of_oz.txt")
BOOK_URL = "https://www.gutenberg.org/files/55/55-0.txt"

# Ensure the directory exists
if not os.path.exists(DATA_DIR):
    print(f"Creating directory: {DATA_DIR}")
    os.makedirs(DATA_DIR)

# Download the book if it doesn't exist
if not os.path.exists(BOOK_PATH):
    print(f"Downloading Wizard of Oz text from {BOOK_URL}...")
    try:
        response = requests.get(BOOK_URL)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(BOOK_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Successfully downloaded and saved to {BOOK_PATH}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        exit(1) # Exit if download fails
    except IOError as e:
        print(f"Error writing file: {e}")
        exit(1) # Exit if writing fails

# Now proceed with reading the file
print(f"Loading text from {BOOK_PATH}...")
with open(BOOK_PATH, encoding="utf-8") as f:
    text = f.read()

# Remove Gutenberg header/footer
text = re.split(r"\*\*\* START OF.*?\*\*\*", text, maxsplit=1)[-1]
text = re.split(r"\*\*\* END OF.*?\*\*\*", text, maxsplit=1)[0]

# Normalize spacing and remove excessive newlines
text = re.sub(r"\s+", " ", text).strip()

# Tokenize into sentences
sentences = sent_tokenize(text)

# Filter sentences
sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 30]
sentences = list(dict.fromkeys(sentences))  # Deduplicate

# ---- Step 2: Generate examples with bit=0 and bit=1 ----
data = []
for sent in sentences:
    for bit in (0, 1):
        data.append({"buffer_text": sent, "bit": bit})

random.seed(42)
random.shuffle(data)

# ---- Step 3: Split and save as HuggingFace dataset ----
n = len(data)
train = data[: int(n * 0.8)]
val = data[int(n * 0.8) : int(n * 0.9)]
test = data[int(n * 0.9) :]

ds = DatasetDict(
    {
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    }
)

ds.save_to_disk("stego_dataset_wizard_of_oz/")
print("✅ Saved dataset with", len(data), "examples")
