# lemmatize_quran.py
import csv
from camel_tools.disambig.mle import MLEDisambiguator

# Load words
with open("quran_words.txt", "r", encoding="utf-8") as f:
    words = [w.strip() for w in f.readlines() if w.strip()]

print("Loaded", len(words), "words.")

# Load CAMeL disambiguator
print("Loading CAMeL MLE disambiguator...")
disambig = MLEDisambiguator.pretrained()

print("Processing...")
results = disambig.disambiguate(words)

# Save to CSV
with open("quran_lemmas.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerow(["word", "lemma"])

    for r in results:
        analysis = r.analyses[0]  # best (top-ranked) analysis
        lemma = analysis.analysis['lex']
        writer.writerow([r.word, lemma])

print("Saved quran_lemmas.csv")
