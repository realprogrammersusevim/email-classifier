# Yes, this is a glorified shell script

import os
import random
import shutil

from tqdm import tqdm

spam_dirs = ["spam", "spam_2"]
ham_dirs = ["easy_ham", "easy_ham_2", "hard_ham"]
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
random.shuffle(spam_files)
random.shuffle(ham_files)

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
        os.system("iconv -c -t ascii dataset/" + dir + "/" + file + " > dataset/" + dir + "/" + file + ".eml")
        os.remove("dataset/" + dir + "/" + file)
