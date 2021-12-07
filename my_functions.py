import numpy as np
import regex as re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)


# Data transformation
def rev_group_list(df, col, grp):
    """
    Function collapses multiple strings into a list of single values based on a group-by column
    :param df: dataframe to group
    :param col: column to group
    :param grp: dataframe grouped at this level
    :return: list
    """
    # drop duplicate data
    df_out = df.drop_duplicates().sort_values(col)

    # Group by grp
    df_out[col] = df_out.groupby([grp])[col].transform(lambda x: '| '.join(x))

    # drop duplicate data
    df_out = df_out.drop_duplicates()

    # Convert to a list
    df_out[col] = df_out[col].apply(lambda x: x.split('| '))

    # Return transformed data frame
    return df_out


def clean_text(x, stop_words_lem=False):
    """
    Function to clean and normalise text
    :param x: text to clean
    :return: string
    """
    # Remove non letters ex white space and dash
    txt = re.sub("[^a-zA-Z -]", "", x)

    # Replace dashes with space
    txt = re.sub("[-]", " ", txt)

    # Remove extra white spaces
    txt = re.sub("\s+", " ", txt)

    # Convert to lower case
    txt = txt.lower()

    # Remove stopwords
    if stop_words_lem:
        txt = remove_stopwords(txt)
        txt = lemmatize_text(txt)

    return txt


def clean_text_list(x, *args):
    """
    Function to apply the clean_text function to elements of a list
    :param x: list of text to clean
    :return: list of text
    """
    lst = []
    for i in x:
        lst.append(clean_text(i, *args))

    return lst


def remove_stopwords(text):
    """
    Function to remove stopwords using nltk stopwords list
    :param text: text to remove stopwords
    :return: string
    """
    # Remove the stop words
    stop_words = set(stopwords.words('english'))
    no_stop_word_text = [w for w in text.split() if not w in stop_words]

    return ' '.join(no_stop_word_text)


def lemmatize_text(text):
    """
    Function to Lemmatize text
    :param text: text to Lemmatize
    :return: cleaned text
    """
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text)


def sid_analyser(x, str_list=True):
    """
    Function to apply SentimentIntensityAnalyzer to elements of a list or string
    :param x: list of text to clean
    :para str_list: True for string (default) and False for list of text
    :return: list of text
    """
    sid = SentimentIntensityAnalyzer()
    if str_list:
        x = sid.polarity_scores(', '.join(x))
    else:
        lst = []
        for i in x:
            lst.append(sid.polarity_scores(i))

        x = lst

    return x


def sentiment_overall(x):
    """
    Function to provide an overall sentiment classification per (Hutto, n.d.)
    :param x: dictionary of polarity_scores
    :return: string (overall sentiment classification)
    """
    if x >= 0.05:
        sent = 'positive'
    elif x <= -0.05:
        sent = 'negative'
    else:
        sent = 'neutral'
    return sent

    # Reference:
    # Hutto, C. J. (n.d.). VADER-Sentiment-Analysis. GitHub.
    # Retrieved November 30, 2021, from https://github.com/cjhutto/vaderSentiment#about-the-scoring


def freq_words_chart(x, terms, title, n=7, m=8):
    """
    Plot of the N (terms) most frequently used words in a corpus
    :param x: Data Frame column (corpus) to plot
    :param terms: N terms to plot
    :param title: chart time (string)
    :return: plot of the most frequently used words
    """
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top N frequent words
    d = words_df.nlargest(columns="count", n=terms)

    # visualize words and frequencies
    plt.figure(figsize=(n, m))
    ax = sns.barplot(data=d, x="count", y="word")
    ax.set(ylabel='Word')
    ax.set(title=title)
    plt.show()
