#!/usr/bin/env python3

import argparse
import json
import logging

import gensim.downloader as api
import nltk
import numpy as np
from nltk.corpus import stopwords


if __name__ == "__main__":
    logging.basicConfig()
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # yapf: disable
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--embeddings", type=str, default="embeddings.npz", help="Where to save the embedded tags/documents")
    parser.add_argument("--idfs", type=str, default="idfs.npz", help="Where to save the IDFs derived from the tags")
    parser.add_argument("--tags", type=str, default="tags.json", help="Where to save the extracted tags")
    args = parser.parse_args()
    # yapf: enable

    log.info(f"loading model and stopwords")
    model = api.load("glove-twitter-25")
    # nltk.download("stopwords")
    stop_words = stopwords.words("english")

    # Dictionaries to be saved
    embeddings = {}
    idfs = {}

    with open(args.tags, "r") as f:
        all_tags = json.load(f)

    log.info(f"gathering corpus")
    for chip_id, tags in all_tags.items():
        tags = " ".join(tags.values())
        tags = tags.replace("_", " ").split()
        tags = map(lambda w: w.lower(), tags)
        tags = list(filter(lambda w: w not in stop_words, tags))
        tags = list(filter(lambda w: w in model, tags))
        all_tags[chip_id] = tags

    log.info("computing inverse document frequencies")
    for chip_id, tags in all_tags.items():
        for word in set(tags):
            idfs[word] = idfs.get(word, 0) + 1
    for word, num_docs_containing_word in idfs.items():
        idfs[word] = np.log(len(all_tags) / (idfs.get(word)))

    log.info("computing term frequencies")
    tf_idfs = {}
    for chip_id, tags in all_tags.items():
        tf_idf = {}
        for word in tags:
            tf_idf[word] = tf_idf.get(word, 0) + 1
        for word in tf_idf:
            tf_idf[word] = tf_idf.get(word) / len(tags)
            tf_idf[word] = tf_idf.get(word) * idfs.get(word)
        tf_idfs[chip_id] = tf_idf

    log.info(f"embedding tags")
    for chip_id, tags in all_tags.items():
        embeddings[chip_id] = []
        for word in tags:
            embeddings.get(chip_id).append(model[word] * tf_idfs.get(chip_id).get(word))
        if len(embeddings.get(chip_id)) == 0:
            embeddings[chip_id] = np.zeros((25,), dtype=np.float32)
        else:
            embeddings[chip_id] = np.mean(embeddings.get(chip_id), axis=0)

    log.info(f"saving to {args.embeddings} and {args.idfs}")
    np.savez_compressed(args.embeddings, **embeddings)
    np.savez_compressed(args.idfs, **idfs)
