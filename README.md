# Email Classifier

Classifying spam using a Transformer Pytorch model. Modern spam filters still
use simple Naive Bayesian classifiers. This project hopes to surpass the
accuracy of these previous models using a Transformer architecture.

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

The `convertcorpus.py` script converts training data from the
[SpamAssasin](https://spamassassin/apache.org/old/publiccorpus),
[CSDMC2010](https://github.com/zrz1996/Spam-Email-Classifier-DataSet), and the
[Fraud Email](https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus?resource=download)
corpuses into a more easily readable form. `spamcorpus.py` contains the class
that turns the converted and cleaned corpus into a Pytorch dataset for training.

## Usage

Once you have the training dataset and you've run the `train.py` script you
should be left with a `model.pth` file that you can use to classify and
categorize spam and ham email.

## TODO

Use the model to classify new emails as they come in and automatically move them
to Junk if they're identified as junk.
