import codecs
import json
from random import shuffle
import numpy as np
import re
from rdflib import Graph
from rdflib.term import Literal, URIRef
from Utilities import *

class EmbeddingGenerator:
    """
    To generate embeddings on sets of triples
    """

    @staticmethod
    def classEmbeddingGeneratorRI(RI_file, instance_types_file,
                                      output_file, renormalize=False, option='simpleSum'):
        """
        We do not
        :param RI_file:
        :param instance_types_file:
        :param prefix_conversion_file
        :param output_file:
        :param renormalize: if True then we will l2-norm each vector in class_embedding_vecs. Otherwise
        not.
        :param option: if simpleSum, will simply add all normalized instance vectors (per class), normalize
        and output. For weightedAverage...
        :return:
        """
        model = read_in_RI_embeddings(RI_file)
        print 'finished reading embeddings file'
        count = 1
        found = 0
        not_found = 0
        class_embedding_vecs = dict()
        with codecs.open(instance_types_file, 'r', 'utf-8') as f:
            for line in f:
                # print 'processing line...',
                # print count
                count += 1
                # if count > 100:
                #     break
                parsed_triple = EmbeddingGenerator.parse_line_into_triple(line)
                if not parsed_triple:
                    print 'line is not triple...',
                    print line
                    continue
                if not parsed_triple['isObjectURI']:
                    print 'object is not a URI...',
                    print line
                    continue
                if parsed_triple['predicate'] != URIRef(u'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
                    print 'predicate is not type...',
                    print line
                    continue
                uri = parsed_triple['subject'].n3()[1:-1]

                try:
                    embedding_vector = l2_normalize(model[uri])
                    found += 1
                    if parsed_triple['object'] not in class_embedding_vecs:
                        class_embedding_vecs[parsed_triple['object']] = np.array([0] * len(embedding_vector))
                    class_embedding_vecs[parsed_triple['object']] = np.sum(
                        [class_embedding_vecs[parsed_triple['object']],
                         embedding_vector], axis=0)
                except KeyError:
                    print 'uri not found in embeddings:',
                    print uri
                    not_found += 1
                    continue
        print 'input file processing complete. URIs found: ',
        print found
        print 'URIs not found: ',
        print not_found
        print 'writing class vecs out to file...'
        if output_file:
            out = codecs.open(output_file, 'w', 'utf-8')
            for k, v in class_embedding_vecs.items():
                answer = dict()
                if not renormalize:
                    answer[k] = v.tolist()
                else:
                    answer[k] = l2_normalize(v)
                json.dump(answer, out)
                out.write('\n')
            out.close()

    @staticmethod
    def classEmbeddingGeneratorDB2Vec(DB2Vec_file, instance_types_file, prefix_conversion_file,
        output_file, renormalize=True, option='simpleAverage'):
        """

        :param DB2Vec_file:
        :param instance_types_file:
        :param prefix_conversion_file
        :param output_file:
        :param renormalize: if True then we will l2-norm each vector in class_embedding_vecs. Otherwise
        not.
        :param option: if simpleAverage, will simply add all instance vectors (per class), normalize
        and output. For weightedAverage...
        :return:
        """
        prefix_dict = dict()
        with codecs.open(prefix_conversion_file, 'r', 'utf-8') as f:
            for line in f:
                fields = re.split('\t', line[0:-1])
                prefix_dict[fields[0]] = fields[1]
        model = Word2Vec.load(DB2Vec_file)
        print 'finished reading embeddings file'
        count = 1
        found = 0
        not_found = 0
        class_embedding_vecs = dict()
        with codecs.open(instance_types_file, 'r', 'utf-8') as f:
            for line in f:
                # print 'processing line...',
                # print count
                count += 1
                # if count > 100:
                #     break
                parsed_triple = EmbeddingGenerator.parse_line_into_triple(line)
                if not parsed_triple:
                    print 'line is not triple...',
                    print line
                    continue
                if not parsed_triple['isObjectURI']:
                    print 'object is not a URI...',
                    print line
                    continue
                if parsed_triple['predicate'] != URIRef(u'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
                    print 'predicate is not type...',
                    print line
                    continue
                uri = parsed_triple['subject'].n3()[1:-1]
                for full_string, prefix in prefix_dict.items():
                    uri = uri.replace(full_string, prefix)
                try:
                    embedding_vector = model[uri]
                    found += 1
                    if parsed_triple['object'] not in class_embedding_vecs:
                        class_embedding_vecs[parsed_triple['object']] = np.array([0] * len(embedding_vector))
                    class_embedding_vecs[parsed_triple['object']] = np.sum([class_embedding_vecs[parsed_triple['object']],
                                                                      embedding_vector], axis=0)
                except KeyError:
                    print 'uri not found in embeddings:',
                    print uri
                    not_found += 1
                    continue
        print 'input file processing complete. URIs found: ',
        print found
        print 'URIs not found: ',
        print not_found
        print 'writing class vecs out to file...'
        if output_file:
            out = codecs.open(output_file, 'w', 'utf-8')
            for k, v in class_embedding_vecs.items():
                answer = dict()
                if not renormalize:
                    answer[k] = v.tolist()
                else:
                    answer[k] = l2_normalize(v)
                json.dump(answer, out)
                out.write('\n')
            out.close()

    @staticmethod
    def classEmbeddingGeneratorDB2Vec_multifile(DB2Vec_file, instance_types_files, prefix_conversion_file,
                                      output_files, instance_cf_file=None, renormalize=False, use_prefix=True, option='simpleAverage'):
        """
        Similar to classEmbedding...DB2Vec but handles multiple instance types files and output files.
        :param DB2Vec_file:
        :param instance_types_files: a list of files.
        :param prefix_conversion_file
        :param output_file:
        :param renormalize: if True then we will l2-norm each vector in class_embedding_vecs. Otherwise
        only the sum gets recorded.
        :param option: if simpleAverage, will simply add all instance vectors (per class), normalize
        and output. For weightedAverage...
        :return:
        """
        instance_cf_dict = dict()
        if option == 'cfAverage' and instance_cf_file:
            with codecs.open(instance_cf_file, 'r', 'utf-8') as f:
                for line in f:
                    fields = line[0:-1].split('\t')
                    instance_cf_dict[fields[0]] = 1.0/int(fields[1])
            print 'finished reading instance cf file...'
            print 'number of instances is...',str(len(instance_cf_dict))
        prefix_dict = dict()
        with codecs.open(prefix_conversion_file, 'r', 'utf-8') as f:
            for line in f:
                fields = re.split('\t', line[0:-1])
                prefix_dict[fields[0]] = fields[1]
        model = Word2Vec.load(DB2Vec_file)
        model.init_sims(replace=True)
        print 'finished reading embeddings file'

        for i in range(len(instance_types_files)):
            instance_types_file = instance_types_files[i]
            print 'processing file...',instance_types_file
            count = 1
            found = 0
            not_found = 0
            class_embedding_vecs = dict()
            with codecs.open(instance_types_file, 'r', 'utf-8') as f:
                for line in f:
                    # print 'processing line...',
                    # print count
                    count += 1
                    # if count > 100:
                    #     break
                    parsed_triple = EmbeddingGenerator.parse_line_into_triple(line)
                    if not parsed_triple:
                        print 'line is not triple...',
                        print line
                        continue
                    if not parsed_triple['isObjectURI']:
                        print 'object is not a URI...',
                        print line
                        continue
                    if parsed_triple['predicate'] != URIRef(u'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
                        print 'predicate is not type...',
                        print line
                        continue
                    uri = parsed_triple['subject'].n3()[1:-1]
                    orig_uri = str(uri)
                    if use_prefix:
                        for full_string, prefix in prefix_dict.items():
                            uri = uri.replace(full_string, prefix)
                    else:
                         uri = '<'+str(uri)+'>'
                    try:
                        embedding_vector = model[uri]
                        if option == 'cfAverage' and instance_cf_dict:
                            if orig_uri not in instance_cf_dict:
                                print orig_uri,' not in instance class frequency dictionary!'
                                continue
                            else:
                                new_embedding_vector = np.multiply(
                            np.array([instance_cf_dict[orig_uri]]*len(embedding_vector)),
                             embedding_vector)
                                embedding_vector = new_embedding_vector
                        found += 1
                        if parsed_triple['object'] not in class_embedding_vecs:
                            class_embedding_vecs[parsed_triple['object']] = np.array([0] * len(embedding_vector))
                        class_embedding_vecs[parsed_triple['object']] = np.sum(
                            [class_embedding_vecs[parsed_triple['object']],
                             embedding_vector], axis=0)
                    except KeyError:
                        # print 'uri not found in embeddings:',
                        # print uri
                        not_found += 1
                        continue
            print 'input file processing complete. URIs found: ',
            print found
            print 'URIs not found: ',
            print not_found
            print 'writing class vecs out to file...'

            out = codecs.open(output_files[i], 'w', 'utf-8')
            for k, v in class_embedding_vecs.items():
                answer = dict()
                if not renormalize:
                    answer[k] = v.tolist()
                else:
                    answer[k] = l2_normalize(v)
                json.dump(answer, out)
                out.write('\n')
            out.close()

    @staticmethod
    def propertyEmbeddingGeneratorDB2Vec(DB2Vec_file, triples_file, prefix_conversion_file,
                                      output_file, renormalize=True, option='simpleAverage'):
        """

        :param DB2Vec_file:
        :param instance_types_file:
        :param prefix_conversion_file
        :param output_file:
        :param renormalize: if True then we will l2-norm each vector in class_embedding_vecs. Otherwise
        not.
        :param option: if simpleAverage, will simply subtract obj - subj, sum, normalize
        and output.
        :return:
        """
        prefix_dict = dict()
        with codecs.open(prefix_conversion_file, 'r', 'utf-8') as f:
            for line in f:
                fields = re.split('\t', line[0:-1])
                prefix_dict[fields[0]] = fields[1]
        model = Word2Vec.load(DB2Vec_file)
        print 'finished reading embeddings file'
        count = 1
        found = 0
        not_found = 0
        property_embedding_vecs = dict()
        with codecs.open(triples_file, 'r', 'utf-8') as f:
            for line in f:
                # print 'processing line...',
                # print count
                count += 1
                # if count > 100:
                #     break
                parsed_triple = EmbeddingGenerator.parse_line_into_triple(line)
                if not parsed_triple:
                    print 'line is not triple...',
                    print line
                    continue
                if not parsed_triple['isObjectURI']:
                    print 'object is not a URI...',
                    print line
                    continue

                subject_uri = parsed_triple['subject'].n3()[1:-1]
                object_uri = parsed_triple['object'].n3()[1:-1]

                for full_string, prefix in prefix_dict.items():
                    subject_uri = subject_uri.replace(full_string, prefix)
                    object_uri = object_uri.replace(full_string, prefix)
                try:
                    embedding_vector_subject = model[subject_uri]
                    embedding_vector_object = model[object_uri]
                    obMinusSub = np.diff([embedding_vector_subject,embedding_vector_object], axis=0)
                    found += 1
                    if parsed_triple['predicate'] not in property_embedding_vecs:
                        property_embedding_vecs[parsed_triple['predicate']] = np.array([0] * len(embedding_vector_subject))
                    property_embedding_vecs[parsed_triple['predicate']] = np.sum(
                        [property_embedding_vecs[parsed_triple['predicate']],
                         obMinusSub[0]], axis=0)
                except KeyError:
                    print 'one of the uris (subject or object) not found in embeddings:',
                    print line
                    not_found += 1
                    continue
        print 'input file processing complete. triples successfully processed: ',
        print found
        print 'triples not processed: ',
        print not_found
        print 'writing property vecs out to file...'
        if output_file:
            out = codecs.open(output_file, 'w', 'utf-8')
            for k, v in property_embedding_vecs.items():
                answer = dict()
                if not renormalize:
                    answer[k] = v.tolist()
                else:
                    answer[k] = l2_normalize(v)
                json.dump(answer, out)
                out.write('\n')
            out.close()

    @staticmethod
    def propertyEmbeddingGeneratorDB2Vec_multifile(DB2Vec_file, triples_files, prefix_conversion_file,
                                         output_files, renormalize=False, option='simpleAverage'):
        """

        :param DB2Vec_file:
        :param instance_types_file:
        :param prefix_conversion_file
        :param output_file:
        :param renormalize: if True then we will l2-norm each vector in class_embedding_vecs. Otherwise
        not.
        :param option: if simpleAverage, will simply subtract obj - subj, sum, normalize
        and output.
        :return:
        """
        prefix_dict = dict()
        with codecs.open(prefix_conversion_file, 'r', 'utf-8') as f:
            for line in f:
                fields = re.split('\t', line[0:-1])
                prefix_dict[fields[0]] = fields[1]
        model = Word2Vec.load(DB2Vec_file)
        model.init_sims(replace=True)
        print 'finished reading embeddings file'
        for i in range(len(triples_files)):
            triples_file = triples_files[i]
            print 'processing file...',triples_file
            count = 1
            found = 0
            not_found = 0
            property_embedding_vecs = dict()
            with codecs.open(triples_file, 'r', 'utf-8') as f:
                for line in f:
                    # print 'processing line...',
                    # print count
                    count += 1
                    # if count > 100:
                    #     break
                    parsed_triple = EmbeddingGenerator.parse_line_into_triple(line)
                    if not parsed_triple:
                        print 'line is not triple...',
                        print line
                        continue
                    if not parsed_triple['isObjectURI']:
                        print 'object is not a URI...',
                        print line
                        continue

                    subject_uri = parsed_triple['subject'].n3()[1:-1]
                    object_uri = parsed_triple['object'].n3()[1:-1]

                    for full_string, prefix in prefix_dict.items():
                        subject_uri = subject_uri.replace(full_string, prefix)
                        object_uri = object_uri.replace(full_string, prefix)
                    try:
                        embedding_vector_subject = model[subject_uri]
                        embedding_vector_object = model[object_uri]
                        obMinusSub = np.diff([embedding_vector_subject, embedding_vector_object], axis=0)
                        found += 1
                        if parsed_triple['predicate'] not in property_embedding_vecs:
                            property_embedding_vecs[parsed_triple['predicate']] = np.array(
                                [0] * len(embedding_vector_subject))
                        property_embedding_vecs[parsed_triple['predicate']] = np.sum(
                            [property_embedding_vecs[parsed_triple['predicate']],
                             obMinusSub[0]], axis=0)
                    except KeyError:
                        # print 'one of the uris (subject or object) not found in embeddings:',
                        # print line
                        not_found += 1
                        continue
            print 'input file processing complete. triples successfully processed: ',
            print found
            print 'triples not processed: ',
            print not_found
            print 'writing property vecs out to file...'
            output_file = output_files[i]
            out = codecs.open(output_file, 'w', 'utf-8')
            for k, v in property_embedding_vecs.items():
                answer = dict()
                if not renormalize:
                    answer[k] = v.tolist()
                else:
                    answer[k] = l2_normalize(v)
                json.dump(answer, out)
                out.write('\n')
            out.close()

    @staticmethod
    def _generate_random_sparse_vector(d, non_zero_ratio):
        """
        Suppose d =200 and the ratio is 0.01. Then there will be 2 +1s and 2 -1s and all the rest are 0s.


        :param d:
        :param non_zero_ratio:
        :return: a numpy array with d dimensions
        """
        answer = [0] * d
        indices = [i for i in range(d)]
        shuffle(indices)
        k = int(non_zero_ratio * d)
        for i in range(0, k):
            answer[indices[i]] = 1
        for i in range(k, 2 * k):
            answer[indices[i]] = -1
        return np.array(answer)

    @staticmethod
    def parse_line_into_triple(line):
        """
        Convert a line into subject, predicate, object, and also a flag on whether object is a literal or URI.
        At present we assume all objects are URIs. Later this will have to be changed.
        :param line:
        :return:
        """
        # fields = re.split('> <', line[1:-2])
        # print fields
        answer = dict()
        g = Graph().parse(data=line, format='nt')
        for s,p,o in g:
            answer['subject'] = s
            answer['predicate'] = p
            answer['object'] = o

        if 'subject' not in answer:
            return None
        else:
            answer['isObjectURI'] = (type(answer['object']) != Literal)
            return answer

    @staticmethod
    def generate_triples_embeddings_v1(triples_file, embedding_output_file, context_output_file, d=500, non_zero_ratio=0.01):
        """
        We use simple arithmetic to learn random indexes.
        """

        context_vecs = dict()
        embedding_vecs = dict()
        literal_context_vecs = dict() # context vectors for tokens that occur in literals
        count = 1
        with codecs.open(triples_file, 'r', 'utf-8') as f:
            for line in f:
                if count%50000==0:
                    print 'processing line...',
                    print count
                count += 1
                # if count > 100:
                #     break
                parsed_triple = EmbeddingGenerator.parse_line_into_triple(line)
                if not parsed_triple:
                    continue
                if parsed_triple['subject'] not in context_vecs:
                    context_vecs[parsed_triple['subject']] = EmbeddingGenerator._generate_random_sparse_vector(d, non_zero_ratio)
                    embedding_vecs[parsed_triple['subject']] = np.array([0]*d)
                if parsed_triple['predicate'] not in context_vecs:
                    context_vecs[parsed_triple['predicate']] = EmbeddingGenerator._generate_random_sparse_vector(d, non_zero_ratio)
                    embedding_vecs[parsed_triple['predicate']] = np.array([0]*d)
                if parsed_triple['isObjectURI']:
                    if parsed_triple['object'] not in context_vecs:
                        context_vecs[parsed_triple['object']] = EmbeddingGenerator._generate_random_sparse_vector(d, non_zero_ratio)
                        embedding_vecs[parsed_triple['object']] = np.array([0]*d)

                    embedding_vecs[parsed_triple['object']] =np.sum([embedding_vecs[parsed_triple['object']],
                        context_vecs[parsed_triple['subject']],context_vecs[parsed_triple['predicate']]], axis=0)
                    # print embedding_vecs[parsed_triple['object']].tolist()
                    obMinusSub = np.diff([context_vecs[parsed_triple['subject']],context_vecs[parsed_triple['object']]], axis=0)
                    # print obMinusSub.tolist()
                    obMinusPred = np.diff([context_vecs[parsed_triple['predicate']], context_vecs[parsed_triple['object']]], axis=0)


                else:
                    continue
                    # literal_embedding = EmbeddingGenerator._generate_embedding_for_literal_object(parsed_triple['object'], literal_context_vecs)
                    # obMinusSub = np.diff(
                    #     [context_vecs[parsed_triple['subject']], literal_embedding], axis=0)
                    # obMinusPred = np.diff(
                    #     [context_vecs[parsed_triple['predicate']], literal_embedding],axis=0)

                embedding_vecs[parsed_triple['subject']] = np.sum(
                    [embedding_vecs[parsed_triple['subject']], obMinusPred[0]], axis=0)
                embedding_vecs[parsed_triple['predicate']] = np.sum(
                    [embedding_vecs[parsed_triple['predicate']], obMinusSub[0]], axis=0)



        # write out embedding vecs to file
        if embedding_output_file:
            out = codecs.open(embedding_output_file, 'w', 'utf-8')
            for k, v in embedding_vecs.items():
                answer = dict()
                answer[k] = v.tolist()
                json.dump(answer, out)
                out.write('\n')
            out.close()

        # # write out context vecs to file
        # if context_output_file:
        #     out = codecs.open(context_output_file, 'w', 'utf-8')
        #     for k, v in context_vecs.items():
        #         answer = dict()
        #         answer[k] = v.tolist()
        #         json.dump(answer, out)
        #         out.write('\n')
        #     out.close()


    @staticmethod
    def generate_triples_embeddings_v2(triples_file, embedding_output_file, context_output_file, d=200, non_zero_ratio=0.01):
        """
        We learn embeddings for 'subjects' based on outgoing context. We assume no literal objects.
        """

        context_vecs = dict()
        # print context_vecs['http://dbpedia.org/ontology/neighboringMunicipality']
        embedding_vecs = dict()
        # literal_context_vecs = dict()  # context vectors for tokens that occur in literals
        count = 1
        with codecs.open(triples_file, 'r', 'utf-8') as f:
            for line in f:
                print 'processing line...',
                print count
                count += 1
                # if count > 100:
                #     break
                parsed_triple = EmbeddingGenerator.parse_line_into_triple(line)
                if parsed_triple['subject'] not in embedding_vecs:
                    embedding_vecs[parsed_triple['subject']] = np.array([0] * d)
                # parsed_triple['object'] = unicode(parsed_triple['object'], 'utf-8')
                # parsed_triple['predicate'] = unicode(parsed_triple['predicate'], 'utf-8')
                # if parsed_triple['predicate'] not in context_vecs or parsed_triple['object'] not in context_vecs:
                #     print 'either predicate or object not in context_vecs for triple:',
                #     print line
                #     print parsed_triple['predicate']
                #     print parsed_triple['object']
                #     print
                #     # break
                #     continue
                # else:
                if parsed_triple['predicate'] not in context_vecs:
                    context_vecs[parsed_triple['predicate']] = EmbeddingGenerator._generate_random_sparse_vector(d, non_zero_ratio)
                if parsed_triple['object'] not in context_vecs:
                    context_vecs[parsed_triple['object']] = EmbeddingGenerator._generate_random_sparse_vector(d, non_zero_ratio)
                obMinusPred = np.diff(
                        [context_vecs[parsed_triple['predicate']], context_vecs[parsed_triple['object']]], axis=0)




                embedding_vecs[parsed_triple['subject']] = np.sum(
                    [embedding_vecs[parsed_triple['subject']], obMinusPred[0]], axis=0)

        # write out embedding vecs to file
        if embedding_output_file:
            out = codecs.open(embedding_output_file, 'w', 'utf-8')
            for k, v in embedding_vecs.items():
                answer = dict()
                answer[k] = v.tolist()
                json.dump(answer, out)
                out.write('\n')
            out.close()

        # write out context vecs to file
        if context_output_file:
            out = codecs.open(context_output_file, 'w', 'utf-8')
            for k, v in context_vecs.items():
                answer = dict()
                answer[k] = v.tolist()
                json.dump(answer, out)
                out.write('\n')
            out.close()

    @staticmethod
    def generate_triples_embeddings_v3(triples_file, embedding_output_file, context_output_file, d=100, non_zero_ratio=0.01):
        """
        We learn embeddings for 'objects' based on incoming context. We assume no literal objects.
        Property information is ignored completely.
        """

        context_vecs = dict()
        embedding_vecs = dict()
        # literal_context_vecs = dict()  # context vectors for tokens that occur in literals
        count = 1
        with codecs.open(triples_file, 'r', 'utf-8') as f:
            for line in f:
                if count%50000==0:
                    print 'processing line...',
                    print count
                count += 1
                # if count > 100:
                #     break
                parsed_triple = EmbeddingGenerator.parse_line_into_triple(line)
                if not parsed_triple:
                    continue
                if not parsed_triple['isObjectURI']:
                    continue
                if parsed_triple['object'] not in embedding_vecs:
                    embedding_vecs[parsed_triple['object']] = np.array([0] * d)
                # parsed_triple['subject'] = unicode(parsed_triple['subject'], 'utf-8')
                # parsed_triple['predicate'] = unicode(parsed_triple['predicate'], 'utf-8')
                # if parsed_triple['predicate'] not in context_vecs or parsed_triple['subject'] not in context_vecs:
                #     print 'either predicate or subject not in context_vecs for triple:',
                #     print line
                #     continue
                # else:
                # if parsed_triple['predicate'] not in context_vecs:
                #     context_vecs[parsed_triple['predicate']] = EmbeddingGenerator._generate_random_sparse_vector(d, non_zero_ratio)
                if parsed_triple['subject'] not in context_vecs:
                    context_vecs[parsed_triple['subject']] = EmbeddingGenerator._generate_random_sparse_vector(d, non_zero_ratio)
                embedding_vecs[parsed_triple['object']] = np.sum(
                [embedding_vecs[parsed_triple['object']],
                 context_vecs[parsed_triple['subject']]], axis=0)



        # write out embedding vecs to file
        if embedding_output_file:
            out = codecs.open(embedding_output_file, 'w', 'utf-8')
            for k, v in embedding_vecs.items():
                answer = dict()
                answer[k] = v.tolist()
                json.dump(answer, out)
                out.write('\n')
            out.close()

        # write out context vecs to file
        if context_output_file:
            out = codecs.open(context_output_file, 'w', 'utf-8')
            for k, v in context_vecs.items():
                answer = dict()
                answer[k] = v.tolist()
                json.dump(answer, out)
                out.write('\n')
            out.close()


# path = '/Users/mayankkejriwal/datasets/eswc2017/dbpedia-experiments/'
# p_path = path+'types-partitions/'
# word2vec_path = '/Users/mayankkejriwal/datasets/heiko-vectors/'
# model = Word2Vec.load(word2vec_path+'DB2VecNoTypes/db2vec_sg_200_5_25_5')
# model.init_sims(replace=True)
# EmbeddingGenerator.propertyEmbeddingGeneratorDB2Vec_multifile(word2vec_path+'DB2Vec_sg_500_5_5_15_4_500',
# [path+'partition-1.ttl',path+'partition-2.ttl',path+'partition-3.ttl',path+'partition-4.ttl',path+'partition-5.ttl'],
#     word2vec_path+'prefix.txt', [path+'propertyVecsSimpleSum1.jl',path+'propertyVecsSimpleSum2.jl',
#                     path+'propertyVecsSimpleSum3.jl',path+'propertyVecsSimpleSum4.jl',path+'propertyVecsSimpleSum5.jl'])
# EmbeddingGenerator.classEmbeddingGeneratorDB2Vec_multifile(word2vec_path+'DB2VecNoTypes/db2vec_sg_200_5_25_5',
# [p_path+'full-partition/partition-1.ttl',p_path+'full-partition/partition-2.ttl',p_path+'full-partition/partition-3.ttl',
#  p_path+'full-partition/partition-4.ttl',p_path+'full-partition/partition-5.ttl'],
#     word2vec_path+'prefix.txt', [p_path+'vectors-type-excl/classVecsCFSum1.jl',p_path+'vectors-type-excl/classVecsCFSum2.jl',
#                     p_path+'vectors-type-excl/classVecsCFSum3.jl',p_path+'vectors-type-excl/classVecsCFSum4.jl',p_path+'vectors-type-excl/classVecsCFSum5.jl'],
#                                     instance_cf_file=path+'instance-cf.tsv', use_prefix=False)
# EmbeddingGenerator.classEmbeddingGeneratorRI(RI_file=path+'embeddings/embedding_vecs_full_v1-500.jl',
#                                             instance_types_file=path+'instance_types_en.ttl',
#                                              output_file=path+'embeddings/RIclassEmbeddings-v1-normSum.jl')
#                                                  triples_file=path+'dbpedia_mappingbased_objects_en.ttl',
#                 prefix_conversion_file=word2vec_path+'prefix.txt', output_file=path+'DB2Vec-propertyVecs-simpleAverage.jl')
# EmbeddingGenerator.generate_triples_embeddings_v1(path+'dbpedia_mappingbased_objects_en.ttl', path+'embedding_vecs_full_v1-500.jl', path+'context_vecs.jl', d=500, non_zero_ratio=0.01)
# EmbeddingGenerator.generate_triples_embeddings_v2(path+'DBpediaTriplesSubset.ttl', path+'embedding_vecs_v2-500.jl', path+'context_vecs.jl', d=500, non_zero_ratio=0.05)
# EmbeddingGenerator.generate_triples_embeddings_v3(path+'dbpedia_mappingbased_objects_en.ttl', path+'embedding_vecs_v3-100.jl', None)
# line = '<http://dbpedia.org/resource/Alabama> <http://dbpedia.org/ontology/largestCity> <http://dbpedia.org/resource/Birmingham,_Alabama> .'
# print EmbeddingGenerator.parse_line_into_triple(line)