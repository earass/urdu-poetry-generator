import spacy
from collections import Counter
import random
import pandas as pd

ur = spacy.blank('ur')


def read_files():
    files = ['ghalib.txt', 'iqbal.txt', 'faiz.txt']
    all_lines = []
    for file in files:
        with open(file, 'r', encoding="utf-8") as text:
            all_lines.extend(text.readlines())
    return all_lines


def tokenize(sentences, backward):
    all_words = []
    for line in sentences:
        sent_tokens = []
        line = line.strip()
        if line:
            # sent_tokens.append('<s>')
            for i in reversed(list(ur(line))):
                sent_tokens.append(str(i))
            # sent_tokens.append('</s>')
        if backward:
            all_words = list(reversed(sent_tokens)) + all_words
        else:
            all_words = sent_tokens + all_words
    exclude_list = ['٪', '!', '%', '`', '‘', '’', '"', ')', '(', '.', ':', "'", '"', '،', '؟']
    all_words = [i for i in all_words if i not in exclude_list]
    return all_words


def get_freq_dict(items):
    return dict(Counter(items))


def get_prob(row, unigram, ngram):
    prob_dict = {}
    comb_words = list(set(row['Combined']))
    for item in comb_words:
        next_word = item.split(' ')[1]
        prob_dict.update({next_word: ngram[item]/unigram[next_word]})
    return prob_dict


def create_n_gram(words, n):
    combs_words = []
    next_prev_list = []
    for i in range(len(words) - n+1):
        c = [words[j] for j in range(i, i+n)]
        c_word = " ".join(c)
        combs_words.append(c_word)
        next_prev_list.append({"NextWord": c[-1], "PrevWords": " ".join(c[:n-1]), "Combined": c_word})

    n_gram = get_freq_dict(combs_words)
    return n_gram, next_prev_list


def generate_model(sentences, n, m_type='standard'):
    if m_type == 'backward':
        all_words = tokenize(sentences, backward=True)
    elif m_type == 'bidirectional':
        all_words = tokenize(sentences, backward=True) + tokenize(sentences, backward=False)
    else:
        all_words = tokenize(sentences, backward=False)
    exclude_probs = ['<s>', '</s>', '‘', '’']
    unigram, _ = create_n_gram(all_words, 1)
    n_gram, next_prev_list = create_n_gram(all_words, n)

    df_words = pd.DataFrame(next_prev_list)
    df_words = df_words.loc[~df_words['PrevWords'].isin(exclude_probs)].groupby(
        ['PrevWords'])['NextWord', 'Combined'].agg(list).reset_index()
    df_words['Prob'] = df_words.apply(lambda x: get_prob(x, unigram, n_gram), axis=1)
    probs = dict(zip(df_words['PrevWords'], df_words['Prob']))
    return probs


def predict_sent(model, length, word):
    sent = [word]
    probs = dict(model[word])
    probs = {k: v for k, v in sorted(probs.items(), key=lambda item: item[1], reverse=True)}
    sent.extend(list(probs.keys())[:length])
    return " ".join(sent)


def predict_next_word(model, word):
    probs = dict(model[word])
    return max(probs, key=probs.get)  # returns word with max prob


def create_poetry(model):
    n_sentences = 12
    count = 0
    for i in range(n_sentences):
        f_word = random.choice(list(model.keys()))
        sent = [f_word]
        sent_length = random.randint(3, 7) - 1
        for w in range(sent_length):
            next_word = predict_next_word(model, f_word)
            sent.append(next_word)
            f_word = next_word
        count += 1
        print(" ".join(sent))
        if count % 4 == 0:
            print(" ")


def execute():
    sentences = read_files()
    standard_model = generate_model(sentences, 2)
    backward_model = generate_model(sentences, 2, m_type='backward')
    bidirectional_model = generate_model(sentences, 2, m_type='bidirectional')
    print("Poetry using standard model")
    create_poetry(standard_model)
    print(" ")
    print("Poetry using backward model")
    create_poetry(backward_model)
    print(" ")
    print("Poetry using bidirectional model")
    create_poetry(bidirectional_model)


if __name__ == '__main__':
    execute()

