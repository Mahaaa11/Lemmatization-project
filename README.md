✔ Project Goal

Arabic Lemmatization using AraBERT & CAMeL-BERT.

✔ Dataset description

Quranic Corpus + additional word-lemma CSV( we maually created this by applying camel tools o the quranic text).

✔ Training pipeline

Preprocessing → label2id mapping → tokenization → alignment → training → evaluation.

✔ Why BERT models

Because contextual Transformers outperform rule-based analyzers in Arabic morphology.

✔ GPU limitations

Explain that 7 epochs with batch size 16 gave best accuracy, but final reproducible version uses:

batch_size = 1

max_length = 64

epochs = 5

fp16 = True

gradient_checkpointing = True

This is the correct and academically honest way to present it.
