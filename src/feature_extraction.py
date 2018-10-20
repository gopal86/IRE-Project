import math
from nltk import ngrams
from tqdm import tqdm


def language_model():
    '''
    - Make a list(list()) that will store the count of n-grams occuring in each
      comment. This list() will later be used to compute the tf-idf feature matrix.
    - Also, we will make a n_grams dict() that will contain all the char n_grams 
      appearing in all the comments.
    '''

    global n_grams_list, n_grams
    n_grams_list = list()
    n_grams = dict()

    # Read the comments.txt file and find the n_grams in each comment.
    comments = open("../train/comments.txt", "r").readlines()

    for comment in comments:
        char_n_grams = list(ngrams(comment[:-1], 3))
        char_n_grams = ["_".join(n_gram) for n_gram in char_n_grams]
        n_grams_list.append(char_n_grams)

        char_n_grams = set(char_n_grams)
        for n_gram in char_n_grams:
            if n_gram in n_grams.keys():
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1

def extract_features():
    '''
    - Character n-grams(trigrams) are used for feature extraction.
    - Build a vector of feature vectors for each of the tweets. 
    - The feature vector for each tweet will be of length (no. of trigrams) 
      with each feature being 0 or 1 marking the presence/absence of the 
      trigram in that particular tweet.
    - So the features matrix will be of dim [(no. of comments) x (no. of trigrams)]
    '''

    global f_matrix
    f_matrix = list()
    D = len(n_grams_list)
    
    for comment_n_grams in tqdm(n_grams_list):
        f_vector = list()
        for n_gram in n_grams.keys():
            # calculate tf-idf of the n_gram and append the tf-idf to the f_vector
            # tf = [ N(occurences of n_gram in comment) / N(occurences of all the n_grams in the comment) ]
            # idf = log { (total no. of comments) / (no. of comments the n_gram has appeared in) }
            tf = (0.1 * comment_n_grams.count(n_gram)) / len(comment_n_grams)
            idf = math.log(D) / n_grams[n_gram]
            w = tf * idf
            f_vector.append(w)

        f_matrix.append(f_vector)

def write_f_matrix():
    f_matrix_writer = open("../train/feature_matrix.txt", "a")
    for f_vector in f_matrix:
        f_vector = [str(f) for f in f_vector]
        f_str = ",".join(f_vector)
        f_matrix_writer.write(str(f_str) + "\n")


if __name__ == "__main__":
    language_model()
    extract_features()
    write_f_matrix()
