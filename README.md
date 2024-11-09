# Email_Classification_Ham_or_Spam

🔍 Project Overview:

In this project, I built a model that can classify emails into two categories: spam or ham. Here's what I did:

Data Collection & Preprocessing:

The dataset I worked with contained labeled emails in the "spam.csv" file.
I preprocessed the data using spaCy to remove stop words, punctuation, and lemmatize the text, ensuring that the model focuses on meaningful features.
Balancing the Dataset:

The dataset was imbalanced, with more ham than spam emails. I downsampled the ham emails to match the number of spam emails, ensuring a balanced model.
Text Tokenization using BERT:

I utilized the BERT tokenizer to process the text data into input format compatible with BERT models.
Model Building with BERT:

Using TensorFlow and Hugging Face Transformers, I fine-tuned a pre-trained BERT model (bert-base-uncased) for binary classification.
I trained the model using the balanced dataset, optimizing it with Adam optimizer and evaluating it with accuracy metrics.
Results & Evaluation:

After training the model for several epochs, the results were quite promising, showing good accuracy in distinguishing spam from ham emails.
The trained model was saved for future use, and I even integrated callbacks to monitor training and improve performance.
🎯 Key Features:

Data Preprocessing: Text cleaning and tokenization for optimal input.
Balanced Dataset: Handling class imbalance to avoid bias.
BERT: Leveraging cutting-edge NLP techniques for accurate classification.
Deployment-ready: Model ready for integration into email systems for spam detection.
🛠 Tools Used:

Python, TensorFlow, BERT, spaCy, Hugging Face Transformers, Google Colab.
