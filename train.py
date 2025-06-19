import argparse
import logging
import os
import re
import time
from collections import Counter, OrderedDict
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import EmailClassifier
from transformer_model import TransformerModel


class Vocab:
    """Custom Vocab class to replace torchtext's."""

    def __init__(self, counter, specials):
        self.counter = counter
        self.specials = specials
        self._token_to_idx = OrderedDict()
        self._idx_to_token = []

        for i, token in enumerate(specials):
            self._token_to_idx[token] = i
            self._idx_to_token.append(token)

        for token, _ in counter.most_common():
            if token not in self._token_to_idx:
                idx = len(self._idx_to_token)
                self._token_to_idx[token] = idx
                self._idx_to_token.append(token)

        self._default_index = -1

    def __len__(self):
        return len(self._idx_to_token)

    def __getitem__(self, token):
        return self._token_to_idx.get(token, self._default_index)

    def __call__(self, tokens):
        return [self[token] for token in tokens]

    def get_itos(self):
        return self._idx_to_token

    def get_stoi(self):
        return self._token_to_idx

    def set_default_index(self, index):
        self._default_index = index


def build_vocab_from_iterator(iterator, specials):
    """Custom build_vocab_from_iterator function to replace torchtext's."""
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    return Vocab(counter, specials)


def basic_english_tokenizer(text):
    text = text.lower()
    text = re.sub(r"([.!?,'/()])", r" \1 ", text)
    return text.split()


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


def train(dataloader, optimizer, epoch, criterion, model, writer):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    total_loss = 0

    for idx, (label, text, extra_arg) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, extra_arg)
        loss = criterion(predicted_label, label)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            global_step = (epoch - 1) * len(dataloader) + idx
            writer.add_scalar("Loss/train", total_loss / log_interval, global_step)
            writer.add_scalar("Accuracy/train", total_acc / total_count, global_step)
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            total_loss = 0
            start_time = time.time()


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, extra_arg) in enumerate(dataloader):
            predicted_label = model(text, extra_arg)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def collate_batch_transformer(batch, pad_idx, max_len=512):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text[:max_len])  # Truncate long sequences

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list_padded = torch.nn.utils.rnn.pad_sequence(
        text_list, batch_first=True, padding_value=pad_idx
    )
    padding_mask = text_list_padded == pad_idx

    return (
        label_list.to(device),
        text_list_padded.to(device),
        padding_mask.to(device),
    )


def predict(model, text, text_pipeline, model_type="embeddingbag", pad_idx=0):
    with torch.no_grad():
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        if model_type == "embeddingbag":
            output = model(processed_text, torch.tensor([0]))
        else:
            processed_text = processed_text.unsqueeze(0)  # Add batch dimension
            mask = processed_text == pad_idx
            output = model(processed_text, mask)
        return output.argmax(1).item() + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--corpus",
        default=None,
        help="Spam corpus to use. If none is provided assumes you used your own emails.",
    )
    parser.add_argument(
        "--model",
        default="embeddingbag",
        choices=["embeddingbag", "transformer"],
        help="Model to use",
    )
    args = parser.parse_args()

    if args.corpus:
        from spamcorpus import Spam
    else:
        from spam import Spam

    writer = SummaryWriter()

    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    logging.info("Loading Tokenizer")
    tokenizer = basic_english_tokenizer
    if args.corpus:
        train_iter = iter(Spam(os.path.join(SCRIPT_DIR, args.corpus), split="train"))
    else:
        train_iter = iter(Spam("dataset.csv", split="train"))

    logging.info("Building Vocab")
    if args.model == "transformer":
        specials = ["<unk>", "<pad>"]
    else:
        specials = ["<unk>"]
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer), specials=specials
    )
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    logging.debug(f"Using device: {device}")

    # num_class = len(set([label for (label, text) in train_iter]))
    num_class = 2
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    emsize = 64
    if args.model == "embeddingbag":
        model = EmailClassifier(vocab_size, emsize, num_class).to(device)
    else:
        # Transformer hyperparameters
        d_model = 128  # embedding dimension
        nhead = 2  # number of heads in multiheadattention
        d_hid = (
            256  # dimension of the feedforward network model in nn.TransformerEncoder
        )
        nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout = 0.2
        model = TransformerModel(
            vocab_size, d_model, nhead, d_hid, nlayers, num_class, dropout
        ).to(device)

    # Hyperparameters
    EPOCHS = 20  # epoch
    LR = 6  # learning rate
    BATCH_SIZE = 64  # batch size for training
    if args.model == "transformer":
        LR = 0.001
        BATCH_SIZE = 16

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
    total_accu = None
    if args.corpus:
        train_iter = Spam(os.path.join(SCRIPT_DIR, args.corpus), split="train")
        test_iter = Spam(os.path.join(SCRIPT_DIR, args.corpus), split="test")
        val_iter = Spam(os.path.join(SCRIPT_DIR, args.corpus), split="val")
    else:
        train_iter = Spam("dataset.csv", split="train")
        test_iter = Spam("dataset.csv", split="test")
        val_iter = Spam("dataset.csv", split="val")

    if args.model == "embeddingbag":
        collate_fn = collate_batch
    else:
        pad_idx = vocab["<pad>"]
        collate_fn = partial(collate_batch_transformer, pad_idx=pad_idx, max_len=512)

    train_dataloader = DataLoader(
        train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        val_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader, optimizer, epoch, criterion, model, writer)
        accu_val = evaluate(valid_dataloader, model, criterion)
        writer.add_scalar("Accuracy/valid", accu_val, epoch)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            if total_accu is None or accu_val > total_accu:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "criterion_state": criterion.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "num_class": num_class,
                        "emsize": emsize if args.model == "embeddingbag" else None,
                        "vocab": vocab,
                        "lr": LR,
                        "model_type": args.model,
                        "model_hyperparams": {
                            "d_model": 128,
                            "nhead": 2,
                            "d_hid": 256,
                            "nlayers": 1,
                            "dropout": 0.2,
                        }
                        if args.model == "transformer"
                        else None,
                    },
                    os.path.join(writer.log_dir, f"model-{args.model}.pth"),
                )
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

        writer.flush()

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader, model, criterion)
    print("test accuracy {:8.3f}".format(accu_test))

    spam_label = {1: "ham", 2: "spam"}

    model = model.to("cpu")
    if args.model == "transformer":
        pad_idx = vocab["<pad>"]
        predict(
            model,
            "This is a test string",
            text_pipeline,
            model_type="transformer",
            pad_idx=pad_idx,
        )
    else:
        predict(model, "This is a test string", text_pipeline)

    # if args.corpus:
    #     for i in ["dataset/1/00000.eml", "dataset/2/0010.eml"]:
    #         with open(os.path.join(SCRIPT_DIR, i), "r") as f:
    #             test_string = f.read()
    #
    #         print(f"This is {spam_label[predict(test_string, text_pipeline)]}")
