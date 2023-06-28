import email
import imaplib
import logging
import os
import time
from os import getenv

import dotenv
import pandas as pd

logging.basicConfig(level=logging.INFO)
config = dotenv.dotenv_values(".env")

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# Initialize the dataset file if it doesn't exist
for dataset in ["spam.csv", "ham.csv"]:
    if not os.path.exists(os.path.join(SCRIPT_DIR, dataset)):
        logging.info(f"Cannot find {dataset}, creating a new one...")
        pd.DataFrame(
            columns=[
                "time",
                "subject",
                "content",
                "receiving_address",
                "sending_address",
                "mime_type",
            ]
        ).to_csv(dataset)


logging.info("Connecting to IMAP server")
imap = imaplib.IMAP4_SSL(config["MAIL_SERVER"])
logging.info("Connected to IMAP server, logging in")
imap.login(config["USERNAME"], config["PASSWORD"])
logging.info("Logged in")

mailbox = "Archive"

for mailbox in ["Junk", "Archive"]:
    imap.select(mailbox)

    status, response = imap.search(None, "ALL")

    if status == "OK":
        # Get the list of message IDs
        message_ids = response[0].split()

        # Count the number of messages
        num_messages = len(message_ids)
        print(f"Number of messages in {mailbox}: {num_messages}")
    else:
        print("Failed to retrieve messages.")

imap.close()
imap.logout()
