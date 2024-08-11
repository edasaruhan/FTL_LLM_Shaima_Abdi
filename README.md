# Introduction
This project aims to develop a sentiment analysis tool that detects and
analyzes public opinions related to climate change, specifically aligned with
Sustainable Development Goal (SDG) 13: Climate Action. In this project,
I aimed to finetune a DistilBERRT Model to make it more proficient in
classifying sentiments within economic texts.


# Model Selection
DistilBERT was selected due to its efficiency and strong performance in
understanding context, which is critical for analyzing climate-related
discussions.

# Data Collection
Data Source: The dataset for this project was collected from [Kaggle].
The data consists of climate change-related.

# Model Implementation
● Loading the Model: The DistilBERT model was loaded from the
Hugging Face model hub using the ‘transformers’library.
The tokenizer associated with DistilBERT was also loaded to
preprocess the text data.
● Sentiment Analysis Implementation: The sentiment analysis
was implemented using the pre-trained DistilBERT model. A
function was created to classify the sentiment of each text entry as
positive, negative, or neutral.


# Fine-Tuning


● Dataset Preparation for Fine-Tuning: A labeled dataset was
created to fine-tune the DistilBERT model for better accuracy in
sentiment detection.
● Fine-Tuning Process: The fine-tuning process involved training
the model on the labeled dataset for a specified number of epochs,
adjusting the learning rate, and monitoring performance using
validation data.


# Climate Change Sentiment Analysis Tool
This project focuses on analyzing public sentiment regarding climate change, leveraging the power of Natural Language Processing (NLP) with the DistilBERT model. The sentiment analysis tool is designed to categorize text data related to climate change into positive, negative, or neutral sentiments, helping to provide insights that align with SDG 13: Climate Action.

# Features
Model: Utilizes the pre-trained distilbert-base-uncased-finetuned-sst-2-english model from Hugging Face.
Sentiment Analysis: Classifies text into positive, negative, or neutral categories.
Custom Dataset: The dataset used for this project was created specifically for analyzing climate change-related sentiments.
Project Structure
data/: Contains the dataset used for training and evaluation.
notebooks/: Jupyter notebooks with the code for data preprocessing, model implementation, and evaluation.
models/: Directory where the fine-tuned model is saved.

# Reference 
SDGs - https://sdgs.un.org/goals

