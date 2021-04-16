# Q1.1
# ------------------
# Write your implementation here.
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words across the corpus
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    # ------------------
    # Write your implementation here.

    corpus_words = set([word for sentence in corpus for word in sentence])
    corpus_words = list(corpus_words)
    corpus_words = sorted(corpus_words)
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words


# ------------------

# Q1.2
# ------------------
# Write your implementation here.
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = {w: i for i, w in enumerate(words)}
    # ------------------
    # Write your implementation here.

    for sentence in corpus:
        for i, word in enumerate(sentence):
            temp = sentence[max(i - window_size, 0):i]
            temp += sentence[i + 1:min(i + window_size + 1, len(sentence))]
            for j, w in enumerate(temp):
                M[word2Ind[w], word2Ind[word]] += 1

    # ------------------

    return M, word2Ind


# ------------------

# Q1.3
# ------------------
# Write your implementation here.
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ------------------
    # Write your implementation here.

    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = svd.fit_transform(M)

    # ------------------

    print("Done.")
    return M_reduced


# ------------------

# Q1.4
# ------------------
# Write your implementation here.
def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2ind.
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , 2)): matrix of 2-dimensioal word embeddings
            word2ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    plt.figure(figsize=(10, 10))
    for idx, w in enumerate(words):
        i = word2Ind[w]
        x = M_reduced[i][0]
        y = M_reduced[i][1]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, w, fontsize=9)
    plt.show()


# ------------------

# Q1.5-Explanation
""" Given that these country names are in top 10 oil exporters list, it makes sense they are cloe together. Also petroleum and industry are close which also makes sense. Even though we see 2 clusters that are true, I think all of these words are close together and have the same context which is energy or petroleum, So it was better that all these words become 1 large cluster, or at least `barrels`, `bpd` and `output` should be clustered together, `energy` shoud join `petroleum` and `industry` and `oil` could join either of the last cluster or `kuwait`, `equador` and `iraq`. """
# ------------------

# Q2.1-Explanation
"""Unlike the previous method, `industry` and `energy` are clustered together which is correct. `equador`, `iraq` and `petroleum` are close together with `oil` a little further away which is fine but `kuwait` is now very far away which doesn't make sense. `output`, `barrels` and `bpd` are still not clustered. The difference is beecause GloVe used wikipedia corpus but the previous model used Reuters crude corpus."""
# ------------------

# Q2.2
# ------------------
# Write your implementation here.
wv_from_bin.most_similar("arms")[:10]  # weapon. # body part.
# ------------------

# Q2.2-Explanation
"""`arms` has 2 meanings, one is a part of body and the other one is in the context of guns and weapons. We can see words from both contexts in the top 10 most similar words, e.g. `hands` vs. `weapons` or `nuclear` and `iraq`. The reason that many of the words have only one of the meanings might be because that meaning of that word in other contexts has occured fewer times and that one meaning has been in the corpus much more frequently, or simply because GloVe is not a **contextualized** word embedding it just picked up one of the meanings from the corpus."""
# ------------------

# Q2.3
# ------------------
# Write your implementation here.
print(f"W1: True W2: Correct W3: False")
print(f"W1,W3: {wv_from_bin.distance('true', 'false')} < W1,W2: {wv_from_bin.distance('true', 'correct')}")
# ------------------

# Q2.3-Explanation
"""I think the reason is because two words might be near each other in the corpus regardless of their meaning, So in this case `true` and `false` have occured more frequently that `true` and `correct` in the corpus, which indicates that in this method meanings and semantics is playing a less significant role than proximity of words."""
# ------------------

# Q2.4-Explanation
"""king - man + woman"""
# ------------------

# Q2.5
# ------------------
# Write your implementation here.
pprint.pprint(wv_from_bin.most_similar(positive=['uk', '$'], negative=['us']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['poland', 'american'], negative=['america']))
# ------------------

# Q2.5-Explanation
"""In my examples, `us:$ :: uk:£` or `america:american :: poland:polish` which are obvious and don't need further explanation."""
# ------------------

# Q2.6
# ------------------
# Write your implementation here.
pprint.pprint(wv_from_bin.most_similar(positive=['accord', 'toyota'], negative=['camry']))
# ------------------

# Q2.6-Explanation
"""`camry:toyota :: accord:honda` but the word vector's top answer is `agreement`."""
# ------------------

# Q2.7
# ------------------
# Write your implementation here.
# Run this cell
# Here `positive` indicates the list of words to be similar to and `negative` indicates the list of words to be
# most dissimilar from.
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'worker'], negative=['man']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'worker'], negative=['woman']))
# ------------------

# Q2.7-Explanation
"""**a.** `employee` is the most similar word to "woman" and "worker" and most dissimilar to "man".

**b.** `workers` is the most similar word to "man" and "worker" and most dissimilar to "woman".

This shows that a man has some people working for him and the woman is working for somebody, meaning that the word vector thinks a woman can not be in charge."""
# ------------------

# Q2.8
# ------------------
# Write your implementation here.
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'doctor'], negative=['man']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'doctor'], negative=['woman']))
# ------------------

# Q2.8-Explanation
"""In my example:

a.`nurse` is most similar to `woman` and `doctor` and the most dissimilar to `man`.

b.`dr.` is most similar to `man` and `doctor` and the most dissimilar to `woman`.

This shows another obvious sign of gender bias, saying that the best a woman can do is to be a nurse and not a doctor and the complete opposite for a man, a man is always a doctor and there's no way a nurse could be a man."""
# ------------------


# Q2.9-Explanation
"""There's no deny in that humans are biased, and that bias finds its way from someone's mind to what they write, and when we train ML models on these texts, it is obvious that they pick up these biases and reflect them in their answers. We could train an ML model to assign different people (man and woman) to different jobs (low-level jobs with minimum to complicated high-level jobs with very high salary) and if the data its trained on is biased, the results will be pretty much predictable."""
# ------------------
