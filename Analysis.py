from Utilities import *

"""
This is for CP2
"""

def analyze_ground_truth_cosine(doc_embeddings14, doc_embeddings17, matches):
    doc14 = read_in_RI_embeddings(doc_embeddings14)
    doc17 = read_in_RI_embeddings(doc_embeddings17)
    list_of_matches = json.load(open(matches))
    site14list = list()
    site17list = list()
    for i in range(len(list_of_matches)):
        if list_of_matches[i]['site_14'] in doc14 and list_of_matches[i]['site_17'] in doc17 \
            and list_of_matches[i]['class']==1:
            site14list.append(list_of_matches[i]['site_14'])
            site17list.append(list_of_matches[i]['site_17'])
        else:
            print 'match pair not processed: ',
            print list_of_matches[i]
    for site14 in site14list:
        vec1 = doc14[site14]
        print site14
        for site17 in site17list:
            print site17+'\t',
            print abs_cosine_sim(vec1, doc17[site17])

def analyze_ground_truth_jaccard(tokens14, tokens17, matches):
    doc14 = read_in_RI_embeddings(tokens14)
    doc17 = read_in_RI_embeddings(tokens17)
    list_of_matches = json.load(open(matches))
    site14list = list()
    site17list = list()
    for i in range(len(list_of_matches)):
        if list_of_matches[i]['site_14'] in doc14 and list_of_matches[i]['site_17'] in doc17 \
            and list_of_matches[i]['class']==1:
            site14list.append(list_of_matches[i]['site_14'])
            site17list.append(list_of_matches[i]['site_17'])
        else:
            print 'match pair not processed: ',
            print list_of_matches[i]
    for site14 in site14list:
        print '\n'
        t14 = set(doc14[site14])
        print site14
        for site17 in site17list:
            print site17+'\t',
            t17 = set(doc17[site17])
            print len(t17.intersection(t14))*1.0/len(t17.union(t14))


# persona = '/Users/mayankkejriwal/datasets/memex-evaluation-november/persona-linking/'
# analyze_ground_truth_cosine(persona+'liberal-doc-embedding-14-idf.json',persona+'liberal-doc-embedding-17-idf.json',
#                      persona + '17_14_matches_training.json')
# analyze_ground_truth_jaccard(persona+'other-files/tokens-14.jl',
#                              persona+'other-files/tokens-17.jl',
#                      persona + '17_14_matches_training.json')