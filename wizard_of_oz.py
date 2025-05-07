import random
import re
from pathlib import Path
from nltk.tokenize import sent_tokenize
from datasets import Dataset, DatasetDict

# ---- Step 1: Load and clean Wizard of Oz text ----
# Make sure you've downloaded the file from Project Gutenberg:
# wget https://www.gutenberg.org/files/55/55-0.txt -O wizard_of_oz.txt

BOOK_PATH = "texts/wizard_of_oz.txt"
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
