import math
import os
import pickle

import spacy
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from tqdm import tqdm

from spamcorpus import Spam

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
NUM_TRAIN = 3000
NUM_TEST = 1000
SPAM_THRESHOLD = 0.75


def save_tokens(tokenizer, data_train, data_test):
    tokens = {}
    total_spam = 0
    total_ham = 0

    for i, (label, text) in tqdm(enumerate(data_train), total=len(data_train)):
        # if i == NUM_TRAIN:
        #     break
        # truncate to first 600,000 characters
        short_text = text[0][:600000]
        tokenized = tokenizer.tokenize(short_text)
        if label[0] == 1:
            total_ham += 1
        else:
            total_spam += 1
        for token in tokenized.tokens:
            if tokens.get(token) is None:
                tokens[token.text] = {"spam": 0, "ham": 0}
            if label[0] == 1:
                tokens[token.text]["ham"] += 1
            else:
                tokens[token.text]["spam"] += 1

    with open(os.path.join(SCRIPT_DIR, "tokens.pkl"), "wb") as f:
        pickle.dump({"tokens": tokens, "spam": total_spam, "ham": total_ham}, f)


def calculate_probabilities(tokens):
    with open(os.path.join(SCRIPT_DIR, "tokens.pkl"), "rb") as f:
        save = pickle.load(f)

    saved_tokens = save["tokens"]
    total_spam = save["spam"]
    total_ham = save["ham"]

    spammy = []

    # print(saved_tokens)

    # \frac{P(token | spam) * P(spam)}{P(token | spam) * P(spam) + P(token | ham) * P(ham)}
    # Calculate in log space to prevent underflow
    for token in tokens:
        saved = saved_tokens.get(token)
        if saved is None:
            continue

        # print(saved)

        p_token_spam = math.log((saved["spam"] + 1) / (total_spam + 2))
        p_spam = math.log(total_spam / (total_spam + total_ham))
        p_token_ham = math.log((saved["ham"] + 1) / (total_ham + 2))
        p_ham = math.log(total_ham / (total_spam + total_ham))

        prob = math.exp(p_token_spam + p_spam) / (
            math.exp(p_token_spam + p_spam) + math.exp(p_token_ham + p_ham)
        )
        spammy.append(prob)

    # take the top 15 most spammy tokens
    # return sorted(spammy, reverse=True)[:15]
    return sorted(spammy, reverse=True)


if __name__ == "__main__":
    train = Spam(os.path.join(SCRIPT_DIR, "dataset"), split="train")
    test = Spam(os.path.join(SCRIPT_DIR, "corpus"), split="test")
    data_train = DataLoader(train)
    data_test = DataLoader(test)
    tokenizer = Tokenizer.from_file("bert_tokenizer.json")

    if not os.path.exists(os.path.join(SCRIPT_DIR, "tokens.pkl")):
        save_tokens(tokenizer, data_train, data_test)

    ham_preds = []
    spam_preds = []
    # for i, (label, text) in tqdm(enumerate(data_test), total=len(data_test)):
    for i, (label, text) in enumerate(data_test):
        # if i == NUM_TEST:
        #     break
        short_text = text[0][:600000]
        tokenized = tokenizer.encode(short_text)

        calc = calculate_probabilities(tokenized.tokens)
        spammy = sum(calc) / len(calc)
        if label[0] == 1:
            # print("Ham: ", spammy)
            ham_preds.append(spammy)
        else:
            # print("Spam: ", spammy)
            spam_preds.append(spammy)

    # Test every threshold from 0.1 to 0.9
    for i in range(10, 100):
        threshold = i / 100
        ham_correct = 0
        spam_correct = 0
        for pred in ham_preds:
            if pred < threshold:
                ham_correct += 1
        for pred in spam_preds:
            if pred >= threshold:
                spam_correct += 1
        ham_accuracy = round((ham_correct / (len(ham_preds))) * 100, 2)

        if ham_accuracy < 95:
            continue

        spam_accuracy = round((spam_correct / (len(spam_preds))) * 100, 2)
        total_accuracy = round(
            (ham_correct + spam_correct) / (len(ham_preds) + len(spam_preds)) * 100, 2
        )
        print(
            f"Threshold: {threshold}, Ham Accuracy: {ham_accuracy}%, Spam Accuracy: {spam_accuracy}%, Total Accuracy: {total_accuracy}%"
        )
