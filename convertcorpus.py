# Yes, this is a glorified shell script

import csv
import os
import random
import shutil
from email.parser import Parser
from email.policy import default

from tqdm import tqdm

# Now convert the other spam email dataset
with open("publiccorpus/conv_fraud.txt") as f:
    lines = f.readlines()

email = []
current_email = []
for line in lines:
    if line.startswith("From r  "):
        current_email.append(line)
        email.append("".join(current_email))
        current_email = []
    else:
        current_email.append(line)

# Write those emails to spam files
os.mkdir("publiccorpus/fraud")
for i, email in enumerate(email):
    with open("publiccorpus/fraud/" + str(i), "w") as f:
        f.write(email)

# Now process the Enron dataset
with open("publiccorpus/enron.csv") as f:
    reader = csv.reader(f)
    enron = list(reader)

# Now write the spam and ham to files
os.mkdir("publiccorpus/yet_more_spam")
os.mkdir("publiccorpus/even_more_ham")
for i, email in enumerate(enron):
    if email[1] == "spam":
        with open("publiccorpus/spam/" + str(i), "w") as f:
            f.write(email[2])
    else:
        with open("publiccorpus/ham/" + str(i), "w") as f:
            f.write(email[2])

spam_dirs = ["more_spam", "spam", "spam_2", "fraud", "yet_more_spam"]
ham_dirs = ["ham", "easy_ham", "easy_ham_2", "hard_ham", "even_more_ham"]
all = [spam_dirs, ham_dirs]

os.mkdir("dataset")
os.mkdir("dataset/1")
os.mkdir("dataset/2")

spam_files = []
ham_files = []

for class_type in all:
    for dir in class_type:
        print("Processing " + dir)
        for file in os.listdir("publiccorpus/" + dir):
            if dir in ham_dirs:
                ham_files.append(dir + "/" + file)
            else:
                spam_files.append(dir + "/" + file)

# Make sure both sets have the same number of files
# random.shuffle(spam_files)
# random.shuffle(ham_files)

# while len(ham_files) > len(spam_files):
#     ham_files.pop()

# Now copy all the files and convert them to ASCII
for i, file in enumerate(ham_files):
    shutil.copy(
        "publiccorpus/" + file,
        "dataset/1/" + str(i).zfill(len(str(len(ham_files)))),
    )

for i, file in enumerate(spam_files):
    shutil.copy(
        "publiccorpus/" + file,
        "dataset/2/" + str(i).zfill(len(str(len(spam_files)))),
    )

for dir in os.listdir("dataset"):
    for file in tqdm(os.listdir("dataset/" + dir)):
        os.system(
            "iconv -c -t ascii dataset/"
            + dir
            + "/"
            + file
            + " > dataset/"
            + dir
            + "/"
            + file
            + ".eml"
        )
        os.remove("dataset/" + dir + "/" + file)


# for dir in os.listdir("dataset"):
#     for file in tqdm(os.listdir("dataset/" + dir)):
#         with open("dataset/" + dir + "/" + file) as f:
#             email = Parser(policy=default).parsestr(f.read())
#         with open("dataset/" + dir + "/" + file, "w") as f:
#             content = email.get_payload()
#
#             if type(content) == list:
#                 content = "".join(content)
#
#             f.write(content)
