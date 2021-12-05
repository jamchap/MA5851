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


def google_scrape(xpath, driver):
    """
    Function to web scrape HTML element from Google search result
    :param xpath: xpath to scrape
    :param driver: webdriver variable
    :return: list of HTML elements
    """
    p = re.compile(r'>(.*?)<')
    l = driver.find_element_by_xpath(xpath)
    a = l.get_attribute('innerHTML')
    a_2 = []

    for i in p.findall(a):
        if i.replace(" ", ""):
            a_2.append(i)

    return a_2


# Cleaning functions
def try_split_gs(x, delim, n):
    """
    Function to extract Google Users score from list
    :param x: list item to transform
    :param delim: delimiter (e.g. %)
    :param n: position in list to extract
    :return: float (should be read as a %)
    """
    try:
        return float(x[0].split(delim)[n]) / 100
    except:
        return np.NaN


def try_split_us(x):
    """
    Function for extracting the Google User Rating Score
    :param x: list item to transform
    :return: float (should be read as a %)
    """
    try:
        return float(x[0]) / 5
    except:
        return np.NaN


def try_split_usrn(x):
    """
    Function to extract the number of reviews from the Google User Ratings
    :param x: list item to transform
    :return: integer
    """
    try:
        return int(x[1].split(' ')[0])
    except:
        return np.NaN


def try_split_md(x):
    """
    Function to extract the movie description from the movie_description list
    :param x: list item to transform
    :return: string
    """
    try:
        return x[1]
    except:
        return np.NaN


def try_split(x, delim, n):
    """
    Function to extract text from list item
    :param x: list to transform
    :param delim: delimiter
    :param n: position in list to extract
    :return: string
    """
    try:
        return x.split(delim)[n]
    except:
        return x


# Clean results of web scraping
def movie_scores(lst_score):
    """
    Function to transform and clean the movie scores from the web scraping
    :param lst_score: list of scores to transform
    :return: Pandas DataFrame
    """
    lst = []
    for i in lst_score:
        lst.append(i[0])

    df_scores = pd.DataFrame(lst)
    df_scores['imdb_score'] = df_scores['IMDb'].apply(lambda x: float(x.split('/')[0]) / float(x.split('/')[1]))
    df_scores['rotten_tom_score'] = df_scores['Rotten Tomatoes'].apply(lambda x: float(try_split(x, '%', 0)) / 100)
    df_scores['metacritic_score'] = df_scores['Metacritic'].apply(lambda x: float(try_split(x, '%', 0)) / 100)

    # Taking the reviews for only IMDb, Rotten Tomatoes and Metacritic
    # The other reviews have too much missing data
    df_scores = df_scores.loc[:, ['Title'
                                     , 'Year'
                                     , 'imdb_score'
                                     , 'rotten_tom_score'
                                     , 'metacritic_score']]

    return df_scores


# Data transformation
def rev_group_list(df, col, grp):
    """
    Function collapses multiple strings into a list of single values based on a group-by column
    :param df: dataframe to group
    :param col: column to group
    :param grp: dataframe grouped at this level (i.e. ISBN)
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


# NLP
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


def freq_words_chart(x, terms, title):
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
    plt.figure(figsize=(7, 8))
    ax = sns.barplot(data=d, x="count", y="word")
    ax.set(ylabel='Word')
    ax.set(title=title)
    plt.show()
