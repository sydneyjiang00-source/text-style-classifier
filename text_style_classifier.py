#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Style Classifier
----------------------
A from-scratch text classification model (no external ML libraries) that builds
a statistical fingerprint of a body of text across five features -- word
frequencies, word lengths, word stems, sentence lengths, and adjacent-word
pairs -- and uses a log-likelihood comparison to decide which of two known
sources an unknown text most resembles.

@author: Sydney Jiang
"""
import math
import ast


def clean_text(txt):
    """Return a list of words from txt with punctuation removed and all
    characters lowercased.

    inputs: txt is a string of text
    """
    for symbol in """.,?"'!;:""":
        # triple quotes let us include both single- and double-quote
        # characters in the set of punctuation to strip
        txt = txt.replace(symbol, '')
    result = txt.lower().split()
    return result


def stem(a):
    """Return the stem of word a using a set of simple suffix rules."""
    special = {'went': 'go', 'could': 'can', 'saw': 'see', 'ate': 'eat',
               'would': 'will', 'had': 'have'}
    if a in special:
        return special[a]

    elif len(a) > 3 and a[-2:] == 'ly':
        return stem(a[:-2])

    elif len(a) > 3 and a[-1] == 'y':
        return stem(a[:-1] + 'i')

    elif len(a) > 3 and a[-2:] == 'es':
        return stem(a[:-2])

    elif len(a) > 3 and a[-1] == 'e':
        return a[:-1]

    elif len(a) > 3 and a[-1] == 's':
        return stem(a[:-1])

    elif len(a) > 4 and a[-3:] == 'est':
        return stem(a[:-3])

    elif len(a) > 4 and a[-2:] == 'ed':
        if a[-3] == a[-4]:
            return stem(a[:-3])
        else:
            return stem(a[:-2])

    elif len(a) > 4 and a[-2:] == 'er':
        if a[-3] == a[-4]:
            return stem(a[:-3])
        else:
            return stem(a[:-2])

    elif len(a) > 5 and a[-3:] == 'ing':
        if a[-4] == a[-5]:
            return stem(a[:-4])
        else:
            return stem(a[:-3])

    elif len(a) > 5 and a[-4:] == 'ment':
        return stem(a[:-4])

    elif len(a) > 5 and a[-4:] == 'tion':
        return stem(a[:-4])

    else:
        return a


def sentence_split(a):
    """Divide the text into individual sentences, splitting on . ? and !"""
    result1 = a.split('.')
    result2 = []
    for a in result1:
        result2 += a.split('?')
    result3 = []
    for b in result2:
        result3 += b.split('!')
    result4 = [c for c in result3 if c != '']
    return result4


def compare_dictionaries(d1, d2):
    """Return the log-similarity score between two feature dictionaries.

    inputs: d1 and d2 are two feature dictionaries
    """
    if d1 == {}:  # d1 is the new (unknown) dictionary
        return -50
    else:
        score = 0
        total = 0
        for a in d1:
            total += d1[a]

        for x in d2:  # d2 is the known dictionary
            if x not in d1:
                p = 0.5 / total
            else:
                p = d1[x] / total
            score += d2[x] * math.log(p)

        return score


class TextModel:
    """A statistical model of a body of text across five features."""

    def __init__(self, model_name):
        """Initialize a new TextModel using the string model_name and set up
        one dictionary per feature."""
        self.name = model_name
        self.words = {}              # word frequency
        self.word_lengths = {}       # word-length frequency
        self.stems = {}              # stem frequency
        self.sentence_lengths = {}   # sentence-length frequency
        self.adjacent_word = {}      # adjacent-word-pair frequency

    def __repr__(self):
        """Return a string with the model name and the size of each feature
        dictionary."""
        s = 'text model name: ' + self.name + '\n'
        s += '  number of words: ' + str(len(self.words)) + '\n'
        s += '  number of word lengths: ' + str(len(self.word_lengths)) + '\n'
        s += '  number of stems: ' + str(len(self.stems)) + '\n'
        s += '  number of sentence lengths: ' + str(len(self.sentence_lengths)) + '\n'
        s += '  number of adjacent words: ' + str(len(self.adjacent_word))

        return s

    def add_string(self, s):
        """Analyze the string s and add its features to all dictionaries."""
        word_list = clean_text(s)
        for w in word_list:
            # word
            if w not in self.words:
                self.words[w] = 1
            else:
                self.words[w] += 1

            # word length
            if len(w) not in self.word_lengths:
                self.word_lengths[len(w)] = 1
            else:
                self.word_lengths[len(w)] += 1

            # stem
            if stem(w) not in self.stems:
                self.stems[stem(w)] = 1
            else:
                self.stems[stem(w)] += 1

        # adjacent word pairs
        for i in range(len(word_list) - 1):
            adjacent = word_list[i] + word_list[i + 1]
            if adjacent not in self.adjacent_word:
                self.adjacent_word[adjacent] = 1
            else:
                self.adjacent_word[adjacent] += 1

        # sentence lengths
        sentence_list = sentence_split(s)
        for v in sentence_list:
            numword = len([w for w in clean_text(v)])
            if numword not in self.sentence_lengths:
                self.sentence_lengths[numword] = 1
            else:
                self.sentence_lengths[numword] += 1

    def add_file(self, filename):
        """Add the contents of the file specified by filename to the model."""
        f = open(filename, 'r', encoding='utf8', errors='ignore')
        a = f.read()
        self.add_string(a)
        f.close()

    def save_model(self):
        """Save the model by writing each feature dictionary to its own file."""
        f = open(self.name + '_' + 'words', 'w')
        f.write(str(self.words))
        f.close()

        f = open(self.name + '_' + 'word_lengths', 'w')
        f.write(str(self.word_lengths))
        f.close()

        f = open(self.name + '_' + 'stems', 'w')
        f.write(str(self.stems))
        f.close()

        f = open(self.name + '_' + 'sentence_lengths', 'w')
        f.write(str(self.sentence_lengths))
        f.close()

        f = open(self.name + '_' + 'adjacent_word', 'w')
        f.write(str(self.adjacent_word))
        f.close()

    def read_model(self):
        """Read the saved dictionaries back from files and assign them to the
        corresponding attributes. Uses ast.literal_eval, which safely parses a
        Python literal without executing arbitrary code (unlike eval)."""
        f = open(self.name + '_' + 'words', 'r')
        d_str = f.read()
        f.close()
        self.words = ast.literal_eval(d_str)

        f = open(self.name + '_' + 'word_lengths', 'r')
        d_str = f.read()
        f.close()
        self.word_lengths = ast.literal_eval(d_str)

        f = open(self.name + '_' + 'stems', 'r')
        d_str = f.read()
        f.close()
        self.stems = ast.literal_eval(d_str)

        f = open(self.name + '_' + 'sentence_lengths', 'r')
        d_str = f.read()
        f.close()
        self.sentence_lengths = ast.literal_eval(d_str)

        f = open(self.name + '_' + 'adjacent_word', 'r')
        d_str = f.read()
        f.close()
        self.adjacent_word = ast.literal_eval(d_str)

    def similarity_scores(self, other):
        """Return a list of log-similarity scores comparing self and other."""
        score = []
        word_score = compare_dictionaries(other.words, self.words)
        score += [word_score]
        word_lengths_score = compare_dictionaries(other.word_lengths, self.word_lengths)
        score += [word_lengths_score]
        stems_score = compare_dictionaries(other.stems, self.stems)
        score += [stems_score]
        sentence_lengths_score = compare_dictionaries(other.sentence_lengths, self.sentence_lengths)
        score += [sentence_lengths_score]
        adjacent_word_score = compare_dictionaries(other.adjacent_word, self.adjacent_word)
        score += [adjacent_word_score]
        return score

    def classify(self, source1, source2):
        """Compare self to two source models and report which source the
        calling text most likely came from."""
        scores1 = self.similarity_scores(source1)
        scores2 = self.similarity_scores(source2)
        print('scores for ' + source1.name + ': ' + str(scores1))
        print('scores for ' + source2.name + ': ' + str(scores2))

        x = 0
        y = 0
        for i in range(len(scores1)):
            if scores1[i] > scores2[i]:
                x += 1
            elif scores1[i] < scores2[i]:
                y += 1
        if x > y:
            print(self.name + ' is more likely to have come from ' + source1.name)
        elif x < y:
            print(self.name + ' is more likely to have come from ' + source2.name)


def test():
    """A minimal sanity check on three short strings."""
    source1 = TextModel('source1')
    source1.add_string('It is interesting that she is interested.')

    source2 = TextModel('source2')
    source2.add_string('I am very, very excited about this!')

    mystery = TextModel('mystery')
    mystery.add_string('Is he interested? No, but I am.')
    mystery.classify(source1, source2)


def run_tests():
    """Build two source models from the speeches of the management teams of
    Cleveland-Cliffs and Delta, then classify four new texts to see which
    company's style each most resembles. (Requires the corresponding .txt
    files in the working directory.)"""
    source1 = TextModel('Cleveland-Cliffs')
    source1.add_file('Cleveland-Cliffs.txt')

    source2 = TextModel('Delta')
    source2.add_file('Delta.txt')

    new1 = TextModel('WR120')
    new1.add_file('WR120 reflection.txt')
    new1.classify(source1, source2)

    new2 = TextModel('Haitian Immigrants study')
    new2.add_file('Haitian Immigrants study.txt')
    new2.classify(source1, source2)

    new3 = TextModel('BU today')
    new3.add_file('BU today.txt')
    new3.classify(source1, source2)

    new4 = TextModel('RMP comments')
    new4.add_file('RMP comments.txt')
    new4.classify(source1, source2)


if __name__ == '__main__':
    test()
