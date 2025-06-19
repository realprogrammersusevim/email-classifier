import csv
import logging
import os
from os import path

import dotenv
from imap_tools import BaseMailBox, MailBox
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
config = dotenv.dotenv_values(".env")

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Initialize the dataset file if it doesn't exist
if not os.path.exists(os.path.join(SCRIPT_DIR, "dataset.csv")):
    logging.info(f"Cannot find dataset.csv, creating a new one...")


def process_mailbox(mailbox: BaseMailBox) -> list[dict]:
    emails = []
    for msg in tqdm(mailbox.fetch()):
        # TODO: Add attachment representation
        email = {
            "subject": msg.subject,
            "from": msg.from_,
            "to": msg.to,
            "content": msg.text or msg.html,
        }
        emails.append(email)

    return emails


def email_str(email: dict) -> str:
    return f"""
From: {email["from"]}
To: {email["to"]}
Subject: {email["subject"]}

{email["content"]}
"""


# get list of email bodies from INBOX folder
mailbox = MailBox(config["MAIL_SERVER"]).login(
    config["USERNAME"], config["PASSWORD"], "INBOX"
)

mailbox.folder.set("Junk")
spam = process_mailbox(mailbox)
mailbox.folder.set("Archive")
ham = process_mailbox(mailbox)

with open(path.join(SCRIPT_DIR, "dataset.csv"), "w") as f:
    writer = csv.writer(f)
    for i in tqdm(spam):
        writer.writerow([1, email_str(i)])
    for i in tqdm(ham):
        writer.writerow([2, email_str(i)])
