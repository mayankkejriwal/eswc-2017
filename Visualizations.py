import tsne
import json
import codecs
from Utilities import *
from matplotlib.pyplot import *


"""
Construct visualization modules for a variety of tasks (not all ESWC related e.g. companies)
"""

def bioInfo_mgiIntact_visualization(doc_embedding):
    """

    :param doc_embedding:
    :return:
    """
    docvecs = read_in_RI_embeddings(doc_embedding)
    X = list()
    labels = list()
    pos_key_limit = 10000
    neg_key_limit = 10000
    pos_key_count = 0
    neg_key_count = 0
    for k, v in docvecs.items():
        key = int(k)
        if key < 0:
            if neg_key_count < neg_key_limit:
                labels.append(0)
                neg_key_count += 1
            else:
                continue
        elif key > 0:
            if pos_key_count < pos_key_limit:
                labels.append(1)
                pos_key_count += 1
            else:
                continue
        else:
            print 'key is 0. We have a problem.'
            continue
        X.append(np.array(v))
    Y = tsne.tsne(np.array(X), 2, 20, 30.0)
    scatter(Y[:, 0], Y[:, 1], 100, labels)
    print 'Total number of items are : ',
    print len(labels)
    show()


def CP1_cluster_visualization(positive_embeddings, negative_embeddings):
    posvecs = read_in_RI_embeddings(positive_embeddings)
    negvecs = read_in_RI_embeddings(negative_embeddings)
    X = list()
    labels = list()
    for k, v in posvecs.items():
        X.append(np.array(v))
        labels.append(1)
    for k, v in negvecs.items():
        X.append(np.array(v))
        labels.append(0)
    Y = tsne.tsne(np.array(X), 2, 20, 30.0)
    scatter(Y[:, 0], Y[:, 1], 100, labels)
    print 'Total number of items are : ',
    print len(labels)
    show()

def CP1_doc_visualization(positive_doc_embeddings, negative_doc_embeddings):
    limit = 100
    posvecs = read_in_RI_embeddings(positive_doc_embeddings)
    negvecs = read_in_RI_embeddings(negative_doc_embeddings)
    X = list()
    labels = list()
    count = 1
    for k, v in posvecs.items():
        if count > limit:
            break
        X.append(np.array(v))
        labels.append(1)
        count += 1
    count = 1
    for k, v in negvecs.items():
        if count > limit:
            break
        X.append(np.array(v))
        labels.append(0)
        count += 1
    Y = tsne.tsne(np.array(X), 2, 20, 30.0)
    scatter(Y[:, 0], Y[:, 1], 100, labels)
    print 'Total number of items are : ',
    print len(labels)
    show()

def companies_visualization(doc_embedding, labels_file):
    """

    :param doc_embedding:
    :param labels_file:
    :return:
    """
    infile = codecs.open(labels_file, 'r', 'utf-8')
    labels_dict = json.load(infile)
    # print obj
    infile.close()
    list_of_clusters = list()
    # all_uris = set()
    for item in labels_dict['groundtruth']:
        cluster = set()
        cluster.add(item['company_url'])
        cluster = cluster.union(set(item['true']))
        list_of_clusters.append(cluster)
    docvecs = read_in_RI_embeddings(doc_embedding)
    uris = docvecs.keys()

    # print uris
    # print len(uris)
    X = list()
    labels = list()
    for uri in uris:
        found_flag = False
        for i in range(0, len(list_of_clusters)):
            for uri_item in list_of_clusters[i]:
                if uri == uri_item:
            # if uri in list_of_clusters[i]:
                    labels.append(i+1)
                    found_flag = True
                    i = len(list_of_clusters)
                    break
        if not found_flag:
            print 'we have a problem; label not found for uri ',
            print uri
            # raise Exception
            continue
        X.append(np.array(docvecs[uri]))
    # print X
    # print labels
    # print len(X)
    # fig, ax = subplots()
    # ax.set_color_cycle(['red', 'black', 'yellow', 'green', 'blue'])
    Y = tsne.tsne(np.array(X), 2, 50, 15.0)
    # y1 = list()
    # y2 = list()
    # colors = ['r', 'b', 'g', 'c', 'y']
    # labels_set = set(labels)
    # for lab in labels_set:
    #     for i in range(len(labels)):
    #         if labels[i] == lab:
    #             y1.append(Y[i,0])
    #             y2.append(Y[i, 1])
    #     ax.scatter(y1, y2, color=colors[-1], marker='^')
    #     show()
    #     colors = colors[0:-1]
    scatter(Y[:, 0], Y[:, 1], 100, labels)
    print 'number of labels is : ',
    print len(list_of_clusters)
    show()



# CP1Path = '/Users/mayankkejriwal/datasets/memex-evaluation-november/CP-1-summer/'
# doc_pos = read_in_RI_embeddings(CP1Path+'positive/external_doc_embeddings.jl')
# doc_neg = read_in_RI_embeddings(CP1Path+'negative/external_doc_embeddings.jl')
# print len(doc_pos.keys())
# print len(doc_neg.keys())
# print len(set(doc_pos.keys()).intersection(set(doc_neg.keys())))
# CP1_doc_visualization(CP1Path+'negative/pos_doc_embeddings.jl',
#                           CP1Path+'negative/native_doc_embeddings.jl')
# companiesTextPath = '/Users/mayankkejriwal/datasets/companies/'
# companies_visualization(companiesTextPath+'gt_doc_embeddings.jl', companiesTextPath+'GT_new.json')
# bioInfoPath = '/Users/mayankkejriwal/datasets/bioInfo/2016-11-08-intact_mgi_comparison/'
# bioInfo_mgiIntact_visualization(bioInfoPath+'mgiIntact_docEmbeddings.jl')

