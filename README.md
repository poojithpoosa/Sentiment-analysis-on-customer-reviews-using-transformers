# Sentiment-analysis-on-customer-reviews-using-Transformer

Any user's comments or reviews on a website or app carry a great deal of weight with the app's developers, the app's consumers, and the app's readers. The shapeless data collected by the sites and applications must be investigated and examined. Sentiment analysis is a well-known and widely appreciated tool for this purpose. Often compared to archaeology or opinion mining, Sentiment Analysis (SA) is a subfield of social science that examines how people feel about various entities (such as events, people, problems, services, companies, and organisations) and the features of those entities. Sentiment analysis of internet reviews is a hot area of study due to the exponential growth of review data. In this project, I have used amazon reviews dataset to performance sentiment analysis. Information gain feature extraction is used to extract the valuable features from the reviews and 2-layer transformer model is implemented for classification. The proposed model gave 99.01% accuracy, 0.025 loss, and 99% precision on the test set and 99.03% accuracy, 0.021 loss, and 99.01% precision on the training set. 

## Aim of the project.

The aim of the project is to analysis the sentiment of the customers on various products of amazon and help customers and businesses to make informed decision. For this task, amazon reviews data is used for sentiment analysis with information gain feature extraction and classification using 2-layer Transformer. The evaluation metrics for the projects are loss and accuracy and precision.


## Research question.
* Does the feature selection help in improving performance of the model?
* Which feature selection techniques is most effective?
* Is Bert transformer improving the performance of the sentiment analysis?

## Methodology

![image](https://user-images.githubusercontent.com/61981756/199016264-93b2b650-f94b-4f8c-9961-79e2c0fa832b.png)

Dataset collection 
For the dataset, I have used amazon reviews which is downloaded from https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews . This dataset includes 1.8 million examples for training and 200,000 test cases for each sentiment. To put it simply, the Amazon reviews dataset is made up entirely of reviews left on the online retail giant. Review data is comprehensive, covering 18 years and 35 million reviews as of March 2013. Plaintext reviews, star ratings, and user feedback are all part of the package. The sentiment for the review is 1 if its negative and 2 if its positive.

Data preprocessing
The success of a NLP model depends on the quality of the data used to construct it, therefore preprocessing is a crucial first step. The data preprocessing step consists of lot of various sub step:
* Converting review into lower case.
* Removing all the digits.
* Removing all punctuations.
* Stripping of white spaces
* Tokenization and padding
* One hot encoding of the sentiment label (output variable)

Feature extraction using information gain.

Dimensionality reduction, of which feature extraction is a subset, is the process of breaking down a large amount of raw data into smaller, more manageable chunks. As a result, processing will be less of a hassle. The abundance of variables is perhaps the most notable feature of these massive data sets. Computing power is needed to process these variables. Because of this, feature extraction is useful for obtaining the most informative aspect of large data sets via the careful selection and combination of variables into features.I have used information gain to remove the unwanted words from the dataset. As the dataset is of reviews, the sentence may contain spelling errors, unknown words etc. by using information gain, I have removed word with does not give any value to the sentiment of the sentence.Methods that use filters are often applied before any significant processing of data is done. characteristics are chosen based on how well they correlate with the result variable in different statistical tests. Here, the idea of an Information gain is very relative. I have selected all the features with information gain more than zero.

Transformer Model

The transformer model has emerged as one of the most significant contributions to the field of deep learning and neural networks in recent years. Its primary function is in high-level natural language processing applications. In the two years after its introduction in 2017, the transformer design has developed and branched out into numerous new variations, applying itself to domains outside language processing. Researchers are always looking for methods to enhance transformers and find new uses for them.

# Proposed model

![image](https://user-images.githubusercontent.com/61981756/199016732-a4b0daa5-5cdf-4777-a000-538c4e9672c9.png)

# Libraries and their version:
•	NumPy (1.21)
•	TensorFlow (2.9)
•	Matplotlib (3.5)
•	String (2.1)
•	Wordcloud (1.8)
•	TextFeatureSelection (0.0.15)
•	Sklearn (1.1.1)
•	Pandas (1.4)

## Results:

Word cloud on negative reviews

![image](https://user-images.githubusercontent.com/61981756/199016965-d41c5d2a-ca90-4930-9c02-f206f4122f36.png)

Word Cloud on positive reviews

![image](https://user-images.githubusercontent.com/61981756/199016997-b0abb5d3-66ef-48b9-b6ee-88089229ec9a.png)

## Analysis of Results 

The model had performed very well and gave excellent results. The model performance is evaluated by using 3 metrics, loss of the model, accuracy, precision.
Accuracy while training.
I have tested the model with feature extraction and without feature extraction. The model with feature extraction showed better results compared to the model without feature extraction. The model with feature extraction gained 5 to 6% performance when compared on accuracy metrics. The accuracy was higher at start of training till end of training.

![image](https://user-images.githubusercontent.com/61981756/199017233-381be51e-40a2-42c2-8182-884d2ad7937c.png)

![image](https://user-images.githubusercontent.com/61981756/199017271-e4e7d259-a326-4244-a4d4-a40fb24df16f.png)

![image](https://user-images.githubusercontent.com/61981756/199017321-84d19049-0b10-49e1-b4a6-581851bf3f71.png)

![image](https://user-images.githubusercontent.com/61981756/199017387-7e817739-814c-42fb-8f9f-a104bf3a3e7f.png)

From the above results I can confirm that the feature extraction using Information gain has the positive effect on the performance of the model. the model performance was increased by around 5 to 6%.





