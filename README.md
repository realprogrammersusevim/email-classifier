# Email Classifier

Classifying spam using a Pytorch model. Modern spam filters still use simple
Naive Bayesian classifiers. This project hopes to surpass the accuracy of these
previous models using a better architecture.

## Setup

Before you do anything you'll need to install the dependencies in the
`requirements.txt` file. I would recommend setting up a Python virtual
environment first to keep this projects required packages separate from those on
your system.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

To train on your own emails rename the `.env.example` file to `.env` and fill
out the values. Run the `email_dataset.py` script to create the database file.
Finally, run `train.py` to train the model.

I have another training dataset that I created from all the available training
data I could find out on the internet. You can find it at
[https://github.com/realprogrammersusevim/email-dataset](https://github.com/realprogrammersusevim/email-dataset).

## Usage

Once you have the training dataset and you've run the `train.py` script you
should be left with a `model.pth` file that you can use to classify and
categorize spam and ham email.

## TODO

Use the model to classify new emails as they come in and automatically move them
to Junk if they're identified as junk.
