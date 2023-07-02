# Email Classifier

Classifying spam using a Transformer Pytorch model. Modern spam filters still
use simple Naive Bayesian classifiers. This project hopes to surpass the
accuracy of these previous models using a Transformer architecture.

## Dataset

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

The goal is to make this a simple app that can automatically download and train
on the user's emails to create a spam classifier fine-tuned specifically for
them.
