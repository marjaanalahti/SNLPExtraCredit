import torch
import pandas as pd
from loaderglove import Lang, ToxicityDataset, translateDatasetEntries, normalizeString, unicodeToAscii, load_glove_embeddings

def test_tokenizer():
    glove_file_path = 'glove.6B.100d.txt'
    glove_embeddings = load_glove_embeddings(glove_file_path)
    
    lang = Lang("eng", glove_embeddings)

    test_sentences = [
        "I couldn’t disagree more with this column; Canadians have moved on and don’t care...",
        "I get the odd feeling Klastri the head of the ACLU of...",
        "”Had”... and is now outed as a hypocrite. "
    ]

    for sentence in test_sentences:
        lang.addSentence(normalizeString(unicodeToAscii(sentence)))

    data = {'id': [1, 2, 3], 'text': test_sentences, 'label': [1, 0, 1]}
    test_df = pd.DataFrame(data)
    test_dataset = ToxicityDataset(test_df, id_col='id', text_col='text', label_col='label', lang=lang)
    
    translated_texts, labels = translateDatasetEntries(test_dataset, lang)

    for original, translated in zip(test_sentences, translated_texts):
        print(f"Translated: {translated}")
        print("-" * 30)
