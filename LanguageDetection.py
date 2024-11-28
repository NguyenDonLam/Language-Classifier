from collections import defaultdict as dd
from math import sqrt
import csv
from collections import Counter
def count_trigrams(document: str) -> dict[str, int]:
    """
    Count the occurrences of each trigram in the input string.

    Args:
        document (str): The input string for which trigrams need to be counted.

    Returns:
        dict: A dictionary where the keys are trigrams and the values are the 
        counts of each trigram in the input string.
    """
    trigrams_dict = Counter(document[i:i+3] for i in range(len(document) - 2))
    return trigrams_dict

def normalise(counts_dict: dict[str, int]) -> dict[str, float]:
    """
        Normalizes the counts of trigrams in a document.

        Args:
            counts_dict (dict): A dictionary where the keys are trigrams and the
            values are the counts of those trigrams in a document.

        Returns:
            dict: A dictionary where the keys are trigrams and the values are 
            the normalized counts of those trigrams in a document. The 
            normalized counts are calculated by dividing each count by the total
            count of all trigrams in the document.

        Example:
            counts_dict = {'abc': 2, 'def': 3, 'ghi': 1}
            normalized_dict = normalise(counts_dict)
            print(normalized_dict)
            # Output: {'abc': 0.2857, 'def': 0.4286, 'ghi': 0.1429}
    """
    mag = sqrt(sum([x**2 for x in counts_dict.values()]))
    return dd(int, {key: value/mag for (key, value) in counts_dict.items()})

def train_classifier(training_set: str)->dd[str, dict[str, float]]:
    """
        Creates a normalized trigram dictionary for all languages in the csv
        file 'training_set'

        Args:
            training_set (str): The name of the csv file to be trained from

        Returns:
            dict: A dictionary of all languages, and their respective normalized
            trigram counts
    """
    with open(training_set, encoding="utf-8") as fp:
        big_dict = dd(dict)
        data_list = csv.reader(fp)
        for language, text in data_list:
            trigrams_dd = count_trigrams(text)
            if big_dict[language]:
                for key in trigrams_dd.keys():
                    big_dict[language][key] += trigrams_dd[key]
            else:
                big_dict[language] = trigrams_dd
        big_dict = {k:normalise(v) for k,v in big_dict.items()}
    return big_dict

default_lang_counts = train_classifier("Train classifier.csv")

def score_document(document: str, 
                   lang_counts: dd[str, dict[str, float]] = 
                   default_lang_counts) -> dict[str, float]:
    """
        Gives the score of each language for the provided document, the higher
        the score, the more the language matches with the document

        Args:
            document (str): The document to be tested agaisnt each different
            languages
            lang_counts (dict): A trained model of each language and their
            normalized trigram count from their specified training data
        
        Returns:
            dict: A dictionary of all languages trained with, and their 
            respective score with respect to the provided document
    """
    
    document_count = count_trigrams(document) #document_trigram_dict
    score_dict = {}
    for language in lang_counts:
        score_dict[language] = sum(map(lambda trigram: document_count[trigram] * lang_counts[language][trigram] 
                                       if trigram in lang_counts[language] else 0, 
                                       document_count))
    return score_dict

def classify_doc(document: str, 
                 lang_counts: dd[str, dict[str, float]] = 
                 default_lang_counts) -> str:
    """
        Determines the language a document is written in basing on a trained
        machine learning model
        
        Args:
            document (str): The name of the document to be read and determined
            lang_counts (dict): A trained machine learning model for each 
            languages
        
        Returns:
            str: The language which matches with the provided document the most
        """
    score_lookup = score_document(document, lang_counts)
    ssl = [(score, language) for language, score in score_lookup.items()]
    ssl.sort(key=lambda x: (-x[0], x[1]))
    if ssl[1][0] > ssl[0][0] - (10 ** (-10)):
        return 'English'
    return ssl[0][1]


g = open("German.txt").read()
print(classify_doc(g))