import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.utils import logging
from torchtext.vocab import build_vocab_from_iterator

from model import EmailClassifier
from spam import Spam

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

logging.info("Loading Tokenizer")
tokenizer = get_tokenizer("basic_english")
train_iter = iter(Spam(os.path.join(SCRIPT_DIR, "dataset"), split="train"))


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


logging.info("Building Vocab")
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
logging.debug(f"Using device: {device}")


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _sample in batch:
        _label, _text = _sample["label"], _sample["text"]
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


train_iter = Spam(os.path.join(SCRIPT_DIR, "dataset"), split="train")
dataloader = DataLoader(
    train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch
)

# num_class = len(set([label for (label, text) in train_iter]))
num_class = 2
vocab_size = len(vocab)
emsize = 64
model = EmailClassifier(vocab_size, emsize, num_class).to(device)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
train_iter = Spam(os.path.join(SCRIPT_DIR, "dataset"), split="train")
test_iter = Spam(os.path.join(SCRIPT_DIR, "dataset"), split="test")
val_iter = Spam(os.path.join(SCRIPT_DIR, "dataset"), split="val")

train_dataloader = DataLoader(
    train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    val_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)


print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))

spam_label = {1: "ham", 2: "spam"}


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


with open("dataset/2/00013.372ec9dc663418ca71f7d880a76f117a", "rb") as f:
    raw_string = f.read()
    test_string = raw_string.decode("utf-8", errors="ignore")

model = model.to("cpu")
print(f"This is {spam_label[predict(test_string, text_pipeline)]}")
