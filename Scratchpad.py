from gensim.models.word2vec import Word2Vec
import gzip
import scipy, math
import numpy
from sklearn import datasets
import re, codecs
import tsne
import pylab
import xmltodict, json
import EmbeddingGenerator
from rdflib.term import Literal, URIRef
from scipy.optimize import minimize


def try1():
    """
    See if you can successfully load the google news word2vec file and print out an output.
    THat's our basic litmus test. it works!
    :return:
    """
    path = '/Users/mayankkejriwal/datasets/eswc2017/disasters/'
    model = Word2Vec.load_word2vec_format(path+'GoogleNews-vectors-negative300.bin', binary=True)
    model.init_sims(replace=True)
    keys = ['charlotte', 'Charlotte', 'yorktown', 'LA']
    for key in keys:
        try:
            # print model.most_similar(positive=['woman', 'king'], negative=['man'])
            j = model[key]
            print 'found...',
            print key
        except KeyError:
            print 'not found...',
            print key
            continue
    print model.similarity('charlotte', 'carolina')
    print model.similarity('LA', 'California')

def try2():
    """
    Read in Heiko's vectors. Finally succeeded, although it takes about a minute to load.
    :return:
    """
    path = '/Users/mayankkejriwal/datasets/heiko-vectors/'
    model = Word2Vec.load(path+'DB2Vec_sg_500_5_5_15_4_500')

    print model['http://purl.org/dc/terms/subject']
    print model['dbo:birthPlace']
    print model['http://dbpedia.org/ontology/birthPlace']
    print len(model)
    print 'success'

def try3():
    """
    try accessing the gz version of freebase. It works!
    :return:
    """
    path = '/Users/mayankkejriwal/datasets/eswc2016/'
    total = 10
    count = 1
    with gzip.open(path+'freebase-rdf-latest.gz', 'rb') as f:
        for line in f:
            print 'line : ',
            print line
            if count > total:
                break
            count += 1

def scipy_try1():
    x0 = [0.01]*10
    bounds = list()
    for i in range(len(x0)):
        bounds.append(tuple([-0.2,1.1]))
    bounds_tuple = tuple(bounds)
    print bounds_tuple
    res = None
    try:
        res = minimize(scipy_fun2, x0, method='L-BFGS-B', bounds=bounds_tuple)
        # res = minimize(scipy_fun2, x0, method='BFGS', bounds=bounds_tuple)
        print res.x
        print scipy_fun2(res.x)
        # print res.nit
        # print res.fun
    except RuntimeError:
        print 'RuntimeError!'


def scipy_fun2(x):
    """
    We'll try to minimize this function in scipy to see what happens. x is a multi-dimensional vector
    :param x:
    :return:
    """
    answer = 0.0
    for i in range(0, len(x)):
        answer += (math.pow(-1,i))*(math.pow((i+1),math.pow(x[i], (i+1))))
    return answer

def scipy_fun1(x):
    """
    We'll try to minimize this function in scipy to see what happens. x is a multi-dimensional vector
    :param x:
    :return:
    """
    answer = 0.0
    for i in range(0, len(x)):
        answer += ((i+1)*math.pow(x[i], (i+1)))
    return answer

def try4():
    """
   To try out t-sne stuff. Works brilliantly.
    :return:
    """
    path = '/Users/mayankkejriwal/git-projects/bioExperiments/tsne_python/'
    mnist = path+'mnist2500_X.txt'
    X = numpy.loadtxt(mnist)
    labels = numpy.loadtxt(path+"mnist2500_labels.txt")
    Y = tsne.tsne(X, 2, 50, 20.0)
    pylab.scatter(Y[:,0], Y[:,1], 20, labels)
    pylab.show()

def try5():
    """
    To try out xmltodict. Seems to be working. Let's use this code for Gully's stuff.
    :return:
    """
    example_xml_file = '/Users/mayankkejriwal/datasets/bioInfo/2016-11-08-intact_mgi_comparison/intact_nxml/1541635.nxml'
    f = codecs.open(example_xml_file, 'r')
    content = f.read()
    f.close()
    print(json.dumps(xmltodict.parse(content), indent=4))

def try6():
    """
    Can we read in sample.ttl and ensure that we handle comment lines and literal lines properly.
    Works well.
    :return:
    """
    sample_file = '/Users/mayankkejriwal/datasets/eswc2017/triples_sample.ttl'
    with codecs.open(sample_file, 'r', 'utf-8') as f:
        for line in f:
            triple_dict = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            if not triple_dict:
                continue
            # print type(triple_dict['object'])
            # print triple_dict
            print triple_dict['subject'].n3()[1:-1]
            # print triple_dict['predicate']==URIRef(u'http://www.w3.org/1999/02/22-rdf-syntax-ns#type')

def try7():
    """
    try accessing the gz version of nyu data.
    :return:
    """
    path = '/Users/mayankkejriwal/datasets/memex-evaluation-november/nyu-text/'
    total = 1
    count = 1
    with gzip.open(path + 'output1.gz', 'rb') as f:
        for line in f:
            print 'line : ',
            print line
            count += 1
            if count > total:
                break


def try8():
    """
    Try reading in the companies ground-truth file. Does it work?
    :return:
    """
    path = '/Users/mayankkejriwal/datasets/companies/'
    gt_file = path+'GT.json'
    infile = codecs.open(gt_file, 'r', 'utf-8')
    obj = json.load(infile)
    print obj
    infile.close()


# scipy_try1()