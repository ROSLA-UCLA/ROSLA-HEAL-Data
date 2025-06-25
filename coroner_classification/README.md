# Rapid Overdose Classification Using Coroner Data

This project contains the code that was written to train and evaluate BERT models and classic ML models on classifying drug overdose data.

The data used in this project conists of the coroner data from David Goodman's [similar work](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2794977) classifying free text inputs, as well as a new

external dataset.

Mlflow needs to be started in the background to log all the information.

``mlflow ui --host 127.0.0.1 -p 5000``

## Sourcing embeddings for CUI2Vec and Glove

In order to embed using the CUI method or the Glove method the embeddings are required for this. These are too large to be uploaded to github so should be downloaded.

[Cui2Vec](https://figshare.com/s/00d69861786cd0156d81?file=10959626)

[Glove](https://nlp.stanford.edu/projects/glove/)

Place these files in the data/required_for_conversion/ directory. The glove.6B.100d.txt version of glove was used.

## Data preparation.

The data preparation has two steps, embedding and outcome manipulation. If you just want to finetune bert and bioclinical bert models then you do not need to do embed so use the noembeddings option in the embedding step. Otherwise pick the appropriate one.

### Embeddings

There are 4 different embedding types used in this project.

To generate each of these embeddings first make sure that the initial data

is in the before_preproc folder in the data folder.

The code also requires 3 files in order to do the CUI conversion and the conversion

to the glove embeddings. These should be called cui2vec_pickled.pkl, df_cui.csv and

glove.6B.100d.txt. These should be in the required_for_conversion directory.

When running the embedding script the script will generate embeddings using glove,

bioclinicalbert, cuis and prep the text for tfidf. These are all used in the single label

and multilabel scripts if just running the BERT finetuning scripts then no embeddings

are required as that is handled within those scripts.

The options are:

- cui
- bioclinicalbert
- tfidf
- glove
- noembeddings

to produce the embeddings use the command:

``python features.py {input_data_file_name} {embedding_type}``

an example of this is for glove embeddings:

``python features.py input_no_processing.csv glove``

### Squashing outcomes

Now we need to make sure we have the right columns by dropping the extra outcomes we don't need and ensuring the 'Others' column is formulated properly. This file should be located in the ``different_embeddings`` folder.

To do this use the dataset.py script. You need to specify process if you want to perform this action.

Only file name the folder structure is implied based on previous commands.

An example of the usage:

``python dataset.py process {input_pickle_file_name}``

``python dataset.py process cui_vec.pkl cui``

This will output our final usable .pkl file to processed_data_cui.pkl

This file will work for each embedding if you want to combine the embeddings to use just one file for all three then run this:

``python dataset.py combine``

Only run this after you have run the other three embedding methods

## Training classic ML models

### Single label

For each file we have we need to call it for each drug / outcome. The file will train 4 models. RandomForest, XGBoost, SVM and Logistic Regression.

The model outputs and evaluations will be stored in MLFlow. mlflow needs to be running in the background for these to work.

To run the training process for the bioclinicalbert embeddings:

``python bioclinicalbert.py {drug}``

these can be changed to for each of the different embedding types e.g.

``python cuis.py methamphetamine``

These are the different outputs in the data:

- "Methamphetamine"
- "Heroin"
- "Cocaine"
- "Fentanyl"
- "Alcohol"
- "Prescription.opioids"
- "Any Opioids"
- "Benzodiazepines"
- "Others"
- "Any Drugs"

### Multi label

The multi label script is run using this command:

``python multi_label.py {model_type}``

an example:

``python multi_label.py XGBoost``

The two model types to choose from are:

- XGBoost
- RandomForest

## Training and evaluating BERT models

### Training

To train the BERT models we use the train.py file in the bert_modeling directory.

This takes the input_location and model_type parameters. The input_location path should be ``processed_data.pkl``.

This is the output of all of our preprocessing steps.

The model_type parameter can be either "BERT" or "bioclinicalbert".

It will default to fine tuning bioclinicalbert.

Usage:

``python train.py {input_data_loc} {model_type}``

And an example of how it is used:

``python train.py processed_data.pkl bioclinicalbert``

The model should be saved under /models/model_type/ in the root directory of this project.

### Evaluation

To evaluate the models run them in a similar manner. This time a boolean value of 1 or 0 is required to specify whether you are evaluating

the external or internal test set. This example uses the internal (0)

``python evaluate.py {input_data_loc} {model_type} {external_or_internal_test}``

``python evaluate.py eval_dataset.csv bioclinicalbert 0``

This should give you a print out of the performance on that dataset.

### Prediction

To predict on a given dataset we need a csv, the bert model we are using and the column that contains the text we are predicting on in the file.

``python {input_data_loc} {text_col} {model_type}``

for example:

``python /path_to_data/pred_data.csv text BERT``
