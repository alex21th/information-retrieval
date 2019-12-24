from __future__ import print_function, division
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import argparse

import numpy as np

def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file:
    :return:
    """

    # Get document terms frequency and overall terms document frequency
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = []
    for (t, w),(_, df) in zip(file_tv, file_df):
        # Modifiquem:
        tfidfw.append([t, w/max_freq*np.log10(dcount/df)])
        #
        pass

    return normalize(tfidfw)

def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    # Modifiquem:
    sum = 0
    for t, w in tw:
        sum += w**2
    sum = np.sqrt(sum)

    for i in range(len(tw)):
        tw[i][1] = tw[i][1]/sum
    #
    return tw

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--nrounds', default=5, type=int, help='Number of applications of Rocchio’s rule')
    parser.add_argument('--k', default=5, type=int, help='Number of top documents considered relevant')
    parser.add_argument('--R', default=3, type=int, help='Maximum terms to be kept in the new query')
    parser.add_argument('--alpha', default=3, type=float, help='Alpha weight in the Rocchio rule')
    parser.add_argument('--beta', default=2, type=float, help='Beta weight in the Rocchio rule')
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()

    index = args.index
    nrounds = args.nrounds
    k = args.k
    R = args.R
    alpha = args.alpha
    beta = args.beta
    query = args.query
    print(f'Input   : {query}')

    try:
        client = Elasticsearch()
        s = Search(using=client, index=index)

        if query is not None:
            dic_query = {}    # Dictionary of first query terms (with weight 1).
            for t in query:
                dic_query[t] = 1

            for round in range(nrounds):
                if len(query) > 0:
                    q = Q('query_string',query=query[0])
                    for i in range(1, len(query)):
                        q &= Q('query_string',query=query[i])

                    s = s.query(q)
                    response = s[0:k].execute()    # We get the k more relevant docs.

                    # We stop iterating if no docs are found.
                    if len(response) == 0:
                        break;

                    copy_response = response

                    # Dictionary of weights of the entire corpus.
                    dictionary = {}
                    for r in response:    # Only returns a specific number of results.
                        list = toTFIDF(client, index, r.meta.id)
                        for t,p in list:
                            if t not in dictionary:
                                dictionary[t] = 0
                            dictionary[t] += p

                    # Creation of the new query q' based on Rocchio's rule.
                    list_new_query = []
                    for t,p in dictionary.items():
                        value = 0
                        if t in dic_query:
                            value = alpha*dic_query[t]+p*beta/k
                        else:
                            value = p*beta/k
                        list_new_query.append([t,value])
                        dic_query[t] = value

                    list_new_query = sorted(list_new_query, key=lambda l:l[1], reverse=True)

                    # Parsing of the new query q'.
                    new_query = []
                    for l in list_new_query[0:R]:
                        new_query.append(l[0] + "^" + str(l[1]))

                    query = new_query
                    print(f'query #{round}: {new_query}')

            try:
                print('')
                for r in copy_response:  # Only returns a specific number of results.
                    print(f'ID= {r.meta.id} SCORE={r.meta.score}')
                    print(f'PATH= {r.path}')
                    print(f'TEXT: {r.text[:50]}')
                    print('-----------------------------------------------------------------')

            except NameError:
                print('No document contains the query.')

        else:
            print('No query parameters passed')

        try:
            print (f"{copy_response.hits.total['value']} Documents")
        except NameError:
            print('The input query is not found in corpus.')

    except NotFoundError:
        print(f'Index {index} does not exists')
