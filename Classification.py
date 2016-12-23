from Utilities import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import tsne
import pylab


class Classification:
    """
    Although I tried to use TokenSupervised, I'm concerned that it's not up to the challenge.
    This could be either due to wrong code or due to hyperparameter optimization.
    This code is being written to address these concerns.
    """

    @staticmethod
    def _cross_validation_experiment_1(vectors_file):
        matrix_dict = Classification.build_data_target_matrix_from_file(vectors_file)
        clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1.0))
        # clf = naive_bayes.MultinomialNB()
        scores = cross_val_score(clf, matrix_dict['data'], matrix_dict['target'], cv=10)
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    @staticmethod
    def _cross_validation_experiment_2(vectors_file1, vectors_file2):
        matrix_dict1 = Classification.build_data_target_matrix_from_file(vectors_file1)
        matrix_dict2 = Classification.build_data_target_matrix_from_file(vectors_file2)
        matrix_dict = Classification.add_data_matrices_and_normalize(matrix_dict1, matrix_dict2)
        clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1.0))
        # clf = naive_bayes.MultinomialNB()
        scores = cross_val_score(clf, matrix_dict['data'], matrix_dict['target'], cv=10)
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    @staticmethod
    def _tsne_visualization_experiment(vectors_file):
        matrix_dict = Classification.build_data_target_matrix_from_file(vectors_file)
        X = np.array(matrix_dict['data'])
        labels = np.array(matrix_dict['target'])
        Y = tsne.tsne(X, 2, 50, 20.0)
        pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
        pylab.show()

    @staticmethod
    def _build_id_matrix(orig_matrix):
        id_matrix = dict()  # id references the index of the list from where it hails
        for i in range(0, len(orig_matrix['id'])):
            id = orig_matrix['id'][i]
            id_matrix[id] = i

        return id_matrix

    @staticmethod
    def add_data_matrices_and_normalize(matrix_1, matrix_2):
        """

        """
        id_matrix_1 = Classification._build_id_matrix(matrix_1)
        id_matrix_2 = Classification._build_id_matrix(matrix_2)
        common_ids = (set(id_matrix_1.keys())).intersection(set(id_matrix_2.keys()))
        answer = dict()
        target = list()
        data = list()
        ids = list()
        for id in common_ids:
            ids.append(id)
            index1 = id_matrix_1[id]
            index2 = id_matrix_2[id]
            data.append(l2_normalize(np.sum([matrix_1['data'][index1], matrix_2['data'][index2]], axis=0)))
            if matrix_1['target'][index1] != matrix_2['target'][index2]:
                raise Exception
            else:
                target.append(matrix_1['target'][index1])
        answer['data'] = data
        answer['target'] = target
        answer['id'] = ids

        return answer

    @staticmethod
    def build_data_target_matrix_from_file(vectors_file):
        """
        Recall that the vectors_file contains three tab-delimited fields in each line. The first field is the id
        the second field is the vector itself, while the third field is the label. We will not preprocess
        the vector any further.
        :param vectors_file:
        :return: A dictionary containing 'id', 'data' and 'target' keys, with the sklearn meanings of the latter two. Since this file is
        only for classification we will ALWAYS assume the target is an integer label.
        """
        target = list()
        data = list()
        id= list()
        with codecs.open(vectors_file, 'r', 'utf-8') as f:
            for line in f:
                line = line[0:-1]
                cols = re.split('\t',line)
                # print cols
                id.append(cols[0])
                target.append(int(cols[2]))
                data.append(convert_string_to_float_list(cols[1]))
        answer = dict()
        answer['data'] = data
        answer['target'] = target
        answer['id'] = id
        return answer


    @staticmethod
    def construct_dbpedia_multi_file_DB2Vec(embeddings_file, raw_tsv_file, output_file, uri_index, label_index, id_index,
                                     process_embeddings,label_dict,prefix_conversion_file):
        """
       If a URI is not found in the embedding_file, we'll print that URI out for posterity.
       Embeddings will not be
       :param embedding_file:
       :param raw_tsv_file:
       :param uri_index:
       :param label_index:
       :param process_embeddings: A function that will be called on each embedding vector, if applicable. Must
       ALWAYS return a list
       For example, use it for l2-normalizing. It's a good idea to use functions from Utilities.
       :param label_dict: a dictionary mapping terms in the label to integers. If we encounter a label
       that is not in the dictionary, we'll raise an exception
       :param prefix_conversion_file:
       :return: None
       """
        prefix_dict = dict()
        with codecs.open(prefix_conversion_file, 'r', 'utf-8') as f:
            for line in f:
                fields = re.split('\t', line[0:-1])
                prefix_dict[fields[0]] = fields[1]
        model = Word2Vec.load(embeddings_file)
        print 'finished reading embeddings file'
        not_found = 0
        found = 0
        header = True
        out = codecs.open(output_file, 'w', 'utf-8')
        with codecs.open(raw_tsv_file, 'r', 'utf-8') as f:
            for line in f:
                if header:
                    header = False
                    continue  # ignore header
                if '\n' not in line and '\r' not in line:
                    fields = re.split('\t', line)
                else:
                    fields = re.split('\t', line[0:-2])
                # print fields
                uri = fields[uri_index]
                for full_string, prefix in prefix_dict.items():
                    uri = uri.replace(full_string, prefix)
                try:
                    embedding_vector = model[uri]
                    found += 1
                    if not process_embeddings:
                        out.write(fields[id_index] + '\t' + str(list(embedding_vector)) + '\t' + str(
                            label_dict[fields[label_index]]) + '\n')
                    else:
                        out.write(
                            fields[id_index] + '\t' + str(process_embeddings(list(embedding_vector))) + '\t' + str(
                                label_dict[fields[label_index]]) + '\n')

                except KeyError:
                    print 'uri not found in embeddings:',
                    print uri
                    not_found += 1
                    continue
        out.close()
        print 'number of uris not found...',
        print not_found
        print 'number of uris found...',
        print found


    @staticmethod
    def construct_dbpedia_multi_file_RI(embeddings_file, raw_tsv_file, output_file, uri_index, label_index, id_index,
                                     process_embeddings,label_dict):
        """
        If a URI is not found in the embedding_file, we'll print that URI out for posterity.
        Embeddings will not be
        :param embedding_file:
        :param raw_tsv_file:
        :param uri_index:
        :param label_index:
        :param process_embeddings: A function that will be called on each embedding vector, if applicable. Must
        ALWAYS return a list
        For example, use it for l2-normalizing. It's a good idea to use functions from Utilities.
        :param label_dict: a dictionary mapping terms in the label to integers. If we encounter a label
        that is not in the dictionary, we'll raise an exception
        :return: None
        """
        full_embeddings = read_in_RI_embeddings(embeddings_file)
        print 'finished reading embeddings file'
        not_found = 0
        found = 0
        header = True
        out = codecs.open(output_file, 'w', 'utf-8')
        with codecs.open(raw_tsv_file, 'r', 'utf-8') as f:
            for line in f:
                if header:
                    header = False
                    continue  # ignore header
                if '\n' not in line and '\r' not in line:
                    fields = re.split('\t', line)
                else:
                    fields = re.split('\t', line[0:-2])
                # print fields
                uri = fields[uri_index]
                if uri not in full_embeddings:
                    print 'uri not found in embeddings:',
                    print uri
                    not_found += 1
                    continue
                else:
                    try:
                        found += 1
                        if not process_embeddings:
                            out.write(fields[id_index] + '\t' + str(full_embeddings[uri]) + '\t' + str(label_dict[fields[label_index]]) + '\n')
                        else:
                            out.write(
                                fields[id_index] + '\t' + str(process_embeddings(full_embeddings[uri])) + '\t' + str(label_dict[fields[label_index]]) + '\n')
                    except KeyError:
                        print 'key error...'
                        print fields
                        continue
        out.close()
        print 'number of uris not found...',
        print not_found
        print 'number of uris found...',
        print found


    @staticmethod
    def feature_vector_script_RI(embedding_file='embedding_vecs_v2-500.jl', output_file='comp-multi-full-norm-v2-500.tsv'):
        dbpedia_path = '/Users/mayankkejriwal/datasets/eswc2017/LOD-ML-data/'
        label_dict = {'bad': 0, 'good': 1}
        Classification.construct_dbpedia_multi_file_RI(dbpedia_path + embedding_file,
                                                       dbpedia_path + 'metacriticMovies/completeDataset.tsv',
                                                       dbpedia_path + 'metacriticMovies/'+output_file,
                                                       uri_index=2, label_index=3, id_index=4,
                                                       process_embeddings=l2_normalize,label_dict=label_dict)
        Classification.construct_dbpedia_multi_file_RI(
           dbpedia_path + embedding_file,
           dbpedia_path + 'metacriticAlbums/FinalDataset.tsv',
           dbpedia_path + 'metacriticAlbums/' + output_file,
           uri_index=-2, label_index=-1, id_index=0,
            process_embeddings=l2_normalize, label_dict=label_dict)
        del label_dict
        label_dict = {'low': 0, 'medium': 1, 'high':2}
        Classification.construct_dbpedia_multi_file_RI(
            dbpedia_path + embedding_file,
            dbpedia_path + 'forbes/completedataset.tsv',
            dbpedia_path + 'forbes/' + output_file,
            uri_index=-2, label_index=-1, id_index=0,
            process_embeddings=l2_normalize, label_dict=label_dict)
        Classification.construct_dbpedia_multi_file_RI(
            dbpedia_path + embedding_file,
            dbpedia_path + 'cities/CompleteDataset.tsv',
            dbpedia_path + 'cities/' + output_file,
            uri_index=2, label_index=-1, id_index=0,
            process_embeddings=l2_normalize, label_dict=label_dict)
        Classification.construct_dbpedia_multi_file_RI(
            dbpedia_path + embedding_file,
            dbpedia_path + 'aaup/CompleteDataset.tsv',
            dbpedia_path + 'aaup/' + output_file,
            uri_index=-4, label_index=-2, id_index=-1,
            process_embeddings=l2_normalize, label_dict=label_dict)



# Classification.feature_vector_script_RI()
# word2vec_path = '/Users/mayankkejriwal/datasets/heiko-vectors/'
# dbpedia_path = '/Users/mayankkejriwal/datasets/eswc2017/LOD-ML-data/metacriticAlbums/'
# Classification._tsne_visualization_experiment(dbpedia_path+'aaup/comp-multi-full-DB2Vec.tsv')
# Classification._cross_validation_experiment_1(dbpedia_path+'comp-multi-full-DB2Vec.tsv')
# Classification._cross_validation_experiment_1(dbpedia_path+'comp-multi-full-norm-v2-500.tsv')
# Classification._cross_validation_experiment_2(dbpedia_path+'comp-multi-full-norm-v2-500.tsv',dbpedia_path+'comp-multi-full-DB2Vec.tsv')


# Classification.construct_dbpedia_multi_file_DB2Vec(word2vec_path+'DB2Vec_sg_500_5_5_15_4_500', dbpedia_path+'metacriticMovies/completeDataset.tsv',
#             dbpedia_path+'metacriticMovies/comp-multi-full-DB2Vec.tsv', uri_index=2, label_index=3, id_index=4,
#         process_embeddings=None,label_dict=label_dict, prefix_conversion_file=word2vec_path+'prefix.txt')