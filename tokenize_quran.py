# tokenize_quran.py
from camel_tools.tokenizers.word import simple_word_tokenize

# Read the cleaned Quran
with open("quran.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

tokens = []
for line in lines:
    # tokenize each verse
    words = simple_word_tokenize(line.strip())
    tokens.extend(words)

# Save all tokens into a file (one word per line)
with open("quran_words.txt", "w", encoding="utf-8") as f:
    for w in tokens:
        f.write(w + "\n")

print("Total tokens:", len(tokens))
print("Saved tokens to quran_words.txt")
