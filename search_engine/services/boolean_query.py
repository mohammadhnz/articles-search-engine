import json
import os.path
import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


class Node:
    def __init__(self, doc_id, freq=None):
        self.freq = freq
        self.doc = doc_id
        self.next_value = None


class WordLinkedList:
    def __init__(self, head=None):
        self.head = head


class BooleanQueryService:
    def __init__(self):
        with open(os.path.dirname(__file__) + '/../../data.json', 'r') as file:
            self.data = json.load(file)
        self.articles = dict()
        self._pre_process()

    def _pre_process(self):
        all_words, stop_words = self._extract_words_from_documents()
        self.unique_words_all = list(set(all_words.keys()))
        self._generate_words_documents_linked_list(stop_words)

    def _extract_words_from_documents(self):
        stop_words = set(stopwords.words('english'))
        all_words = {}
        for article in self.data:
            words = self._extract_words_from_article(article, stop_words)
            all_words.update(self._finding_all_unique_words_and_freq(words))
        return all_words, stop_words

    def _finding_all_unique_words_and_freq(self, words):
        return {word: words.count(word) for word in list(set(words))}

    def _extract_words_from_article(self, article, stop_words):
        text = article['title']
        authors = article['authors']
        for auther in authors:
            text += " " + auther['name']
        text = self._remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        words = word_tokenize(text)
        words = [word.lower() for word in words if len(words) > 1 and word not in stop_words]
        return words

    def _remove_special_characters(self, text):
        regex = re.compile('[^a-zA-Z0-9\s]')
        text_returned = re.sub(regex, '', text)
        return text_returned

    def _generate_words_documents_linked_list(self, stop_words):
        linked_list_data = {}
        for word1 in self.unique_words_all:
            linked_list_data[word1] = WordLinkedList()
            linked_list_data[word1].head = Node(1, Node)
        doc_index = 1
        for article in self.data:
            words = self._extract_words_from_article(article, stop_words)
            word_freq_in_doc = self._finding_all_unique_words_and_freq(words)
            for word in word_freq_in_doc.keys():
                linked_list = linked_list_data[word].head
                while linked_list.next_value is not None:
                    linked_list = linked_list.next_value
                linked_list.next_value = Node(doc_index, word_freq_in_doc[word])
            self.articles[doc_index] = article
            doc_index += 1
        self.linked_list_data = linked_list_data

    def query(self, query, result_count=3):
        query = word_tokenize(query)
        connecting_words, words = self._split_query(query)
        total_articles = len(self.articles.keys())
        zeroes_and_ones_of_all_words = self._extract_zeroes_and_ones_of_all_words(total_articles, words)

        for word in connecting_words:
            word_list1 = zeroes_and_ones_of_all_words[0]
            word_list2 = zeroes_and_ones_of_all_words[1]
            if word == "and":
                bitwise_op = [w1 & w2 for (w1, w2) in zip(word_list1, word_list2)]
                zeroes_and_ones_of_all_words.remove(word_list1)
                zeroes_and_ones_of_all_words.remove(word_list2)
                zeroes_and_ones_of_all_words.insert(0, bitwise_op);
            elif word == "or":
                bitwise_op = [w1 | w2 for (w1, w2) in zip(word_list1, word_list2)]
                zeroes_and_ones_of_all_words.remove(word_list1)
                zeroes_and_ones_of_all_words.remove(word_list2)
                zeroes_and_ones_of_all_words.insert(0, bitwise_op);
            elif word == "not":
                bitwise_op = [not w1 for w1 in word_list2]
                bitwise_op = [int(b == True) for b in bitwise_op]
                zeroes_and_ones_of_all_words.remove(word_list2)
                zeroes_and_ones_of_all_words.remove(word_list1)
                bitwise_op = [w1 & w2 for (w1, w2) in zip(word_list1, bitwise_op)]
        # zeroes_and_ones_of_all_words.insert(0, bitwise_op);

        articles = []
        lis = zeroes_and_ones_of_all_words[0]
        cnt = 1
        for index in lis:
            if index == 1:
                articles.append(self.articles[cnt])
            cnt = cnt + 1
        return self._represent_docs(articles[:result_count])

    def _split_query(self, query):
        connecting_words = []
        words = []
        for word in query:
            if word.lower() != "and" and word.lower() != "or" and word.lower() != "not":
                words.append(word.lower())
            else:
                connecting_words.append(word.lower())
        return connecting_words, words

    def _extract_zeroes_and_ones_of_all_words(self, total_articles, words):
        zeroes_and_ones_of_all_words = []
        for word in words:
            if word.lower() in self.unique_words_all:
                zeroes_and_ones = [0] * total_articles
                linkedlist = self.linked_list_data[word].head
                while linkedlist.next_value is not None:
                    zeroes_and_ones[linkedlist.next_value.doc - 1] = 1
                    linkedlist = linkedlist.next_value
                zeroes_and_ones_of_all_words.append(zeroes_and_ones)
            else:
                zeroes_and_ones = [0] * total_articles
                zeroes_and_ones_of_all_words.append(zeroes_and_ones)
        return zeroes_and_ones_of_all_words

    def _represent_docs(self, results):
        return [
            {'paper_title': item['title'], 'paper_id': item['paperId']}
            for item in results
        ]


boolean_query_service = BooleanQueryService()
