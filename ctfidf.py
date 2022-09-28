import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, tokenize, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, tokenizer=tokenize).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=4):
    words = count.get_feature_names()
    w = count.transform(docs_per_topic.content_summary.values).toarray()
    labels = list(docs_per_topic.label)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    top_n_words_cnt = {}
    for i, _ in enumerate(labels):
        top_n_words_cnt['{}'.format(i)] = {}
        for j, word in enumerate(words):
            top_n_words_cnt['{}'.format(i)]['{}'.format(word)] = w[i][j]
    return top_n_words