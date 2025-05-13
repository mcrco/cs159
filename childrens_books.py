import random
import re
import os
import requests
from nltk.tokenize import sent_tokenize
from datasets import Dataset, DatasetDict
import nltk

# Ensure NLTK's sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)

# ---- Configuration for Children's Classics ----
BOOKS_DATA = [
    {"title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "url": "https://www.gutenberg.org/files/11/11-0.txt", "filename": "alice_in_wonderland.txt"},
    {"title": "Through the Looking-Glass", "author": "Lewis Carroll", "url": "https://www.gutenberg.org/files/12/12-0.txt", "filename": "through_the_looking_glass.txt"},
    {"title": "The Wonderful Wizard of Oz", "author": "L. Frank Baum", "url": "https://www.gutenberg.org/files/55/55-0.txt", "filename": "wizard_of_oz.txt"},
    {"title": "Peter Pan (Peter and Wendy)", "author": "J.M. Barrie", "url": "https://www.gutenberg.org/files/16/16-0.txt", "filename": "peter_pan.txt"},
    {"title": "The Secret Garden", "author": "Frances Hodgson Burnett", "url": "https://www.gutenberg.org/files/17396/17396-0.txt", "filename": "the_secret_garden.txt"},
    {"title": "A Little Princess", "author": "Frances Hodgson Burnett", "url": "https://www.gutenberg.org/files/146/146-0.txt", "filename": "a_little_princess.txt"},
    {"title": "Anne of Green Gables", "author": "L. M. Montgomery", "url": "https://www.gutenberg.org/files/45/45-0.txt", "filename": "anne_of_green_gables.txt"},
    {"title": "Black Beauty", "author": "Anna Sewell", "url": "https://www.gutenberg.org/files/271/271-0.txt", "filename": "black_beauty.txt"},
]

DATA_DIR = "texts/childrens_classics/"
OUTPUT_DATASET_DIR = "stego_dataset_childrens_classics/"

# Ensure the download directory exists
if not os.path.exists(DATA_DIR):
    print(f"Creating directory: {DATA_DIR}")
    os.makedirs(DATA_DIR)

all_sentences = []

for book in BOOKS_DATA:
    book_path = os.path.join(DATA_DIR, book["filename"])
    
    # Download the book if it doesn't exist
    if not os.path.exists(book_path):
        print(f"Downloading {book['title']} from {book['url']}...")
        try:
            response = requests.get(book["url"])
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(book_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Successfully downloaded and saved to {book_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {book['title']}: {e}")
            continue # Skip this book if download fails
        except IOError as e:
            print(f"Error writing {book['title']} to file: {e}")
            continue # Skip this book if writing fails
    else:
        print(f"{book['title']} already downloaded at {book_path}")

    # Proceed with reading and processing the file
    print(f"Loading and processing text from {book_path}...")
    try:
        with open(book_path, encoding="utf-8") as f:
            text = f.read()

        # Remove Gutenberg header/footer
        text = re.split(r"\*\*\* START OF TH(?:IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, maxsplit=1, flags=re.IGNORECASE)[-1]
        text = re.split(r"\*\*\* END OF TH(?:IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, maxsplit=1, flags=re.IGNORECASE)[0]
        
        # Normalize spacing and remove excessive newlines
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize into sentences
        sentences = sent_tokenize(text)

        # Filter sentences
        current_book_sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 30]
        all_sentences.extend(current_book_sentences)
        print(f"Found {len(current_book_sentences)} suitable sentences in {book['title']}.")

    except Exception as e:
        print(f"Error processing {book['title']}: {e}")

# Deduplicate all collected sentences
print(f"Total sentences before deduplication: {len(all_sentences)}")
unique_sentences = sorted(list(dict.fromkeys(all_sentences))) # Sort for consistency
print(f"Total unique sentences after deduplication: {len(unique_sentences)}")

# ---- Generate examples with bit=0 and bit=1 ----
data = []
for sent in unique_sentences:
    for bit in (0, 1):
        data.append({"buffer_text": sent, "bit": bit})

random.seed(42)
random.shuffle(data)

# ---- Split and save as HuggingFace dataset ----
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
print(f"✅ Saved combined dataset with {len(data)} examples to {OUTPUT_DATASET_DIR}")
