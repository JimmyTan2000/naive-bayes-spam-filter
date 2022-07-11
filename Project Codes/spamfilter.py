import pandas as pd

# reading the data from the dataset 
data_original = pd.read_csv('SMSSpamCollection', header=None, sep='\t', names=['Label', 'SMS'])

"""
# summary of the dataset
# displaying the first few row of the data set
print(data_original.head())
# represent the dimensionality of the data
print(data_original.shape)
# grouping the label of the dataset
print(data_original.groupby('Label').count())
"""

# cleaning the data
# copy the original data to another set of data
data_clean = data_original.copy()
# remove all the punctuations
data_clean['SMS'] = data_clean['SMS'].str.replace('\W+', ' ', regex = True)\
    .str.replace('\s+', ' ', regex = True).str.strip()
# setting all the characters to lower case
data_clean['SMS'] = data_clean['SMS'].str.lower()
# split it into separate words
data_clean['SMS'] = data_clean['SMS'].str.split()

"""
#checking the cleaned data set
print(data_clean.head())

# calculate the percentage of spam and ham in this dataset
print(data_clean['Label'].value_counts()/ data_original.shape[0] * 100)

"""

#splitting training and test data
train_data = data_clean.sample(frac=0.8, random_state=1).reset_index(drop=True)
test_data = data_clean.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

"""

#checking the distribution of train data
print(train_data.shape)
print(train_data['Label'].value_counts()/ train_data.shape[0] * 100)

#checking distribution of test_data
print(test_data['Label'].value_counts() / test_data.shape[0] * 100)
print(test_data.shape)
print(test_data.head())
"""

# all the words in the sms of the train data
all_words_train = train_data['SMS'].sum()

# remove repeated words and put them into a list
all_vocabs_train = list(set(all_words_train))

# Calculate frequencies of every words for each message
word_counts_per_sms = pd.DataFrame([
    [row[1].count(word) for word in all_vocabs_train]
    for _, row in train_data.iterrows()], columns=all_vocabs_train)

train_data = pd.concat([train_data.reset_index(), word_counts_per_sms], axis=1).iloc[:,1:]

"""
print(train_data.head())
print(train_data.columns)
"""

# Implementing the naive bayes classifier with Lapacian Smoothing
# This model will compute smoothed (k = 1) probabilities 
k = 1
number_vocabulary = len(all_vocabs_train)
probability_spam = (k + train_data['Label'].value_counts()['spam']) / ((k * 2) + train_data.shape[0]) 
probability_ham = (k + train_data['Label'].value_counts()['ham']) /((k * 2) + train_data.shape[0])
number_words_spam = train_data.loc[train_data['Label'] == 'spam', 'SMS'].apply(len).sum()
number_words_ham = train_data.loc[train_data['Label'] == 'ham', 'SMS'].apply(len).sum()

def check_algorithm():
    return probability_ham + probability_spam == 1 \
    and number_words_ham + number_words_spam == len(all_words_train)

def probability_spam_word(word):
    if word in all_vocabs_train:
        return (train_data.loc[train_data['Label'] == 'spam', word].sum() + k) \
            / (number_words_spam + (k * number_vocabulary))
    else:
        return k / (number_words_spam + (k * number_vocabulary))

def probability_ham_word(word):
    if word in all_vocabs_train:
        return (train_data.loc[train_data['Label'] == 'ham', word].sum() + k) \
            / (number_words_ham + (k * number_vocabulary))
    else:
        return k / (number_words_ham + (k * number_vocabulary))

# Classifier for spam filter
def classify(message):
    relative_probability_spam = probability_spam
    relative_probability_ham = probability_ham
    for word in message:
        relative_probability_spam *= probability_spam_word(word)
        relative_probability_ham *= probability_ham_word(word)
    if relative_probability_spam > relative_probability_ham:
        return "spam"
    elif relative_probability_ham > relative_probability_spam:
        return "ham"
    else:
        return "requires human classification"

test_data['predicted'] = test_data['SMS'].apply(classify)

print(test_data.head())

def true_positive():
    tp = 0
    for i in range(len(test_data)):
        if test_data.iloc[i]["Label"] == 'spam' \
            and test_data.iloc[i]["predicted"] == 'spam':
            tp += 1
    return tp

def false_positive():
    fp = 0
    for i in range(len(test_data)):
        if test_data.iloc[i]["Label"] == 'ham' \
            and test_data.iloc[i]["predicted"] == "spam":
            fp += 1
    return fp

def true_negative():
    tn = 0
    for i in range(len(test_data)):
        if test_data.iloc[i]["Label"] == 'ham' \
            and test_data.iloc[i]["predicted"] == 'ham':
            tn += 1
    return tn

def false_negative():
    fn = 0
    for i in range(len(test_data)):
        if test_data.iloc[i]["Label"] == 'spam' \
            and test_data.iloc[i]["predicted"] == 'ham':
            fn += 1
    return fn

def precision():
    return true_positive()/(false_positive() + true_positive())

def accuracy():
    return (true_positive() + true_negative())\
        /(true_positive() + false_negative() + true_negative() + false_positive())

def recall():
    return (true_positive())/(false_negative() + true_positive())

def f1():
    return (2 * precision() * recall())/(precision() + recall())

print(f"Precision score: {precision()}\nAccuracy score: {accuracy()}\n\
Recall score: {recall()}\nF1 score: {f1()}")
