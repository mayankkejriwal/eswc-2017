from Utilities import *
import EmbeddingGenerator
from rdflib.term import URIRef
import random
import tsne
from matplotlib.pyplot import *
import time
from thread import start_new_thread

"""
Specifically designed for some ad-hoc setup for the eswc 2017 embedding paper. Code is not
meant for re-use beyond submitting the paper.
"""

def dbpedia_typification_baseline(types_partition_files, DB2Vec_file, prefix_file, output_files, k=10):
    """

    :param types_partition_file:
    :param DB2Vec_file:
    :param prefix_file:
    :param output_file:
    :param k:
    :return:
    """
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    model = Word2Vec.load(DB2Vec_file)
    model.init_sims(replace=True)
    print 'finished reading embeddings file'
    for i in range(len(types_partition_files)):
        types_partition_file = types_partition_files[i]
        found = 0
        not_found = 0
        out = codecs.open(output_files[i], 'w')
        with codecs.open(types_partition_file, 'r', 'utf-8') as f:
            for line in f:
                try:
                    parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
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
                    # orig = str(uri)
                    for full_string, prefix in prefix_dict.items():
                        uri = uri.replace(full_string, prefix)
                    try:
                        # embedding_vector = model[uri_rep]
                        sims = model.most_similar(positive=[uri],topn=k)
                        found += 1
                        if not sims:
                            print 'no similar vectors found.'
                            continue
                        else:
                            out.write(uri)
                            for item in sims:
                                try:
                                    out.write('\t'+item[0]+'\t'+str(item[1]))
                                except Exception as e:
                                    print 'exception inside sims for loop...'
                                    print e
                                    continue
                            out.write('\n')
                        # out.write(line)
                    except Exception as e:
                        # print 'uri not found in embeddings:',
                        # print uri
                        print e
                        print uri
                        # print sims
                        print 'Error in word embedding...'
                        not_found += 1
                        continue
                except:
                    print 'Exception! continuing...'
                    continue
        out.close()


def dbpedia_types_statistics(pruned_types_file, output_file):
    """
    output_file is a tab delimited file that contains a type, tab, the type's count of instances
    :param pruned_types_file:
    :param output_file:
    :return:
    """
    type_dict = dict()
    count = 1
    with codecs.open(pruned_types_file, 'r', 'utf-8') as f:
        for line in f:
            # if count > 100:
            #     break
            count += 1
            try:
                parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
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
                # instance_uri = parsed_triple['subject'].n3()[1:-1]
                class_uri = parsed_triple['object'].n3()[1:-1]
                if class_uri not in type_dict:
                    type_dict[class_uri] = 0
                type_dict[class_uri] += 1
            except:
                print 'Exception! continuing...'
                continue
    out = codecs.open(output_file, 'w', 'utf-8')
    keys = type_dict.keys()
    keys.sort()
    for k in keys:
        out.write(str(k)+'\t'+str(type_dict[k])+'\n')
    out.close()


def partition_file(input_file, partition_folder, num_partitions=5):
    outs = list()
    for i in range(num_partitions):
        outs.append(codecs.open(partition_folder+'partition-'+str(i+1)+'.ttl','w','utf-8'))
    # out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as f:
        for line in f:
            k = random.randint(1, 5)
            outs[k-1].write(line)
    for i in range(num_partitions):
        outs[i].close()


def prune_dbpedia_types_file(instance_types_file, DB2Vec_file, prefix_file, output_file):
    """
    We need to prune the instance-types file so that it only contains valid type-triples statements
    such that the instance actually exists in the DB2Vec file.
    :param instance_types_file:
    :param DB2Vec_file:
    :param prefix_file:
    :param output_file:
    :return:
    """
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    model = Word2Vec.load(DB2Vec_file)
    model.init_sims(replace=True)
    print 'finished reading embeddings file'
    count = 0
    found = 0
    not_found = 0
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(instance_types_file, 'r', 'utf-8') as f:
        for line in f:
            # print 'processing line...',
            # print count
            count += 1
            if count % 10000==0:
                print 'processing line...',count
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
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
                out.write(line)
            except KeyError:
                # print 'uri not found in embeddings:',
                # print uri
                not_found += 1
                continue
    out.close()
    print 'input file processing complete. URIs found: ',
    print found
    print 'URIs not found: ',
    print not_found

def random_baseline_analysis(exp_partition_file, partition_file, ignoreThing=True):
    """

    :param exp_partition_file:
    :param partition_file:
    :param ignoreThing:
    :return:
    """
    exp1_dict = dict()
    with codecs.open(exp_partition_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                exp1_dict[k] = v
    ranked_dict = dict()
    for k, v in exp1_dict.items():
        elements = v.keys()
        random.shuffle(elements)
        ranked_dict[k] = elements

    # count = 0
    mrr = list()
    with codecs.open(partition_file, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            label = parsed_triple['object'].n3()[1:-1]
            if ignoreThing:
                if 'owl#Thing' in label:
                    continue
            instance = parsed_triple['subject'].n3()[1:-1]
            try:
                rank = ranked_dict[instance].index(label)
                # print rank
                rank += 1
                mrr.append(1.0 / rank)
                # count += 1
            except Exception as e:
                # count += 1
                print e
                mrr.append(0)
                continue

    # mrr /= count
    print '95% confidence intervals for random baseline mrr is...',
    print mean_confidence_interval(mrr)[1:3]


def zero_shot_analysis(zero_shot_dict_file, output_file=None):
    """
    Be careful. the data structures here are very different from those in the other analyses functions.

    :param zero_shot_dict_file:
    :param partition_file:
    :return:
    """
    semantic_code_matrix = list()
    class_labels = list()
    with codecs.open(zero_shot_dict_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                if k == 'label':
                    class_labels.append(v)
                else:
                    types = v.keys()
                    types.sort()
                    code = list()
                    for t in types:
                        code.append(v[t])
                    code.sort()
                    semantic_code_matrix.append(code)
    y_labels = convert_string_labels_to_integers(class_labels)
    Y = tsne.tsne(l2_normalize_matrix(np.array(semantic_code_matrix)), 2, 20, 10.0)
    if output_file:
        out = codecs.open(output_file, 'w', 'utf-8')
        for i in range(len(semantic_code_matrix)):
            answer = dict()
            answer['orig_x']=semantic_code_matrix[i]
            answer['tsne_x']=Y[i].tolist()
            answer['integer_label']=y_labels[i]
            answer['type_label']=class_labels[i]
            json.dump(answer, out)
            out.write('\n')
        out.close()
    scatter(Y[:, 0], Y[:, 1], 100, y_labels)
    print 'Total number of items are : ',
    print len(y_labels)
    show()

def zero_shot_analysis_binary(zero_shot_dict_file, output_file=None):
    """
    Be careful. the data structures here are very different from those in the other analyses functions.
    For this one, we will assign 'known' to class 1 and 'unknown' to class 0
    :param zero_shot_dict_file:
    :param partition_file:
    :return:
    """
    semantic_code_matrix = list()
    class_labels = list()
    with codecs.open(zero_shot_dict_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)

            for k, v in obj.items():
                if k == 'label':
                    continue
                    # class_labels.append(v)
                else:
                    types = v.keys()
                    types.sort()
                    code = list()
                    for t in types:
                        code.append(v[t])
                    if obj['label'] in types:
                        class_labels.append(1)
                    else:
                        class_labels.append(0)
                    code.sort()
                    semantic_code_matrix.append(code)

    # y_labels = convert_string_labels_to_integers(class_labels)
    y_labels = class_labels
    Y = tsne.tsne(l2_normalize_matrix(np.array(semantic_code_matrix)), 2, 20, 10.0)
    if output_file:
        out = codecs.open(output_file, 'w', 'utf-8')
        for i in range(len(semantic_code_matrix)):
            answer = dict()
            answer['orig_x']=semantic_code_matrix[i]
            answer['tsne_x']=Y[i].tolist()
            answer['integer_label']=y_labels[i]
            answer['type_label']=class_labels[i]
            json.dump(answer, out)
            out.write('\n')
        out.close()
    scatter(Y[:, 0], Y[:, 1], 100, y_labels)
    print 'Total number of items are : ',
    print len(y_labels)
    show()


def convert_string_labels_to_integers(string_label_list):
    conversion_dict = dict()
    count = 1
    answer = list()
    for string in string_label_list:
        if string in conversion_dict:
            answer.append(conversion_dict[string])
        else:
            conversion_dict[string] = count
            count += 1
            answer.append(conversion_dict[string])
    return answer


def exp_recall_analysis(exp_partition_file, partition_file, output_file, ignoreThing=False):
    """
    for recall at k. We'll let k range from 1 to 10
    :param exp_partition_file:
    :param partition_file:
    :param ignoreThing:
    :return:
    """
    exp1_dict = dict()
    with codecs.open(exp_partition_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                exp1_dict[k] = v
    ranked_dict = dict()
    for k, v in exp1_dict.items():
        ranked_dict[k] = obtain_ranked_list_from_score_dict(v)
    count = 0
    rankAtK_dict = dict()
    for i in range(1, 416):
        rankAtK_dict[i] = 0
    with codecs.open(partition_file, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            label = parsed_triple['object'].n3()[1:-1]
            if ignoreThing:
                if 'owl#Thing' in label:

                    continue
            instance = parsed_triple['subject'].n3()[1:-1]
            try:
                rank = ranked_dict[instance].index(label)
                rank += 1
                # if rank > 10:
                #     count += 1
                #     continue
                # if rank not in ranked_dict:
                #     ranked_dict[rank] = 0
                rankAtK_dict[rank] += 1
                # mrr.append(1.0/rank)
                count += 1
            except Exception as e:
                count += 1
                # print e
                # mrr.append(0)
                continue
    kys = rankAtK_dict.keys()
    # print rankAtK_dict
    kys.sort()
    # print kys
    for i in range(len(kys)-1):
        val = rankAtK_dict[kys[i]]
        rankAtK_dict[kys[i+1]] += val
        # for j in range(i+1, len(kys)):
        #     rankAtK_dict[kys[j]] += val
    for k in rankAtK_dict.keys():
        rankAtK_dict[k] = (rankAtK_dict[k]*1.0/count)
    out = codecs.open(output_file, 'w', 'utf-8')
    out.write('k,recall\n')
    for k in kys:
        out.write(str(k)+','+str(rankAtK_dict[k])+'\n')
    out.close()


def exp_recall_analysis_big_partition(exp_partition_file, partition_file, output_file, ignoreThing=False):
    """
    for recall at k. We'll let k range from 1 to 10
    :param exp_partition_file:
    :param partition_file:
    :param ignoreThing:
    :return:
    """
    # exp1_dict = dict()
    ranked_dict = dict()
    with codecs.open(exp_partition_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                ranked_dict[k] = obtain_ranked_list_from_score_dict(v)
    print 'finished reading in exp file'
    # for k, v in exp1_dict.items():
    #     ranked_dict[k] = obtain_ranked_list_from_score_dict(v)
    count = 0
    rankAtK_dict = dict()
    for i in range(1, 416):
        rankAtK_dict[i] = 0
    with codecs.open(partition_file, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            label = parsed_triple['object'].n3()[1:-1]
            if ignoreThing:
                if 'owl#Thing' in label:

                    continue
            instance = parsed_triple['subject'].n3()[1:-1]
            try:
                rank = ranked_dict[instance].index(label)
                rank += 1
                # if rank > 10:
                #     count += 1
                #     continue
                # if rank not in ranked_dict:
                #     ranked_dict[rank] = 0
                rankAtK_dict[rank] += 1
                # mrr.append(1.0/rank)
                count += 1
            except Exception as e:
                count += 1
                # print e
                # mrr.append(0)
                continue
    kys = rankAtK_dict.keys()
    # print rankAtK_dict
    kys.sort()
    # print kys
    for i in range(len(kys)-1):
        val = rankAtK_dict[kys[i]]
        rankAtK_dict[kys[i+1]] += val
        # for j in range(i+1, len(kys)):
        #     rankAtK_dict[kys[j]] += val
    for k in rankAtK_dict.keys():
        rankAtK_dict[k] = (rankAtK_dict[k]*1.0/count)
    out = codecs.open(output_file, 'w', 'utf-8')
    out.write('k,recall\n')
    for k in kys:
        out.write(str(k)+','+str(rankAtK_dict[k])+'\n')
    out.close()

def exp_mrr_analysis(exp_partition_file, partition_file, ignoreThing=True):
    exp1_dict = dict()
    with codecs.open(exp_partition_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                exp1_dict[k] = v
    ranked_dict = dict()
    for k, v in exp1_dict.items():
        ranked_dict[k] = obtain_ranked_list_from_score_dict(v)
    # count = 0
    mrr = list()
    with codecs.open(partition_file, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            label = parsed_triple['object'].n3()[1:-1]
            if ignoreThing:
                if 'owl#Thing' in label:

                    continue
            instance = parsed_triple['subject'].n3()[1:-1]
            try:
                rank = ranked_dict[instance].index(label)
                rank += 1
                mrr.append(1.0/rank)
                # count += 1
            except Exception as e:
                # count += 1
                # print e
                mrr.append(0)
                continue

    # mrr /= count
    # print '95% confidence intervals for mrr is...',
    print str(mean_confidence_interval(mrr)[0])+'\t'+str(mean_confidence_interval(mrr)[1])


def obtain_ranked_list_from_score_dict(score_dict):
    reversed_dict = reverse_dict(score_dict)
    keys = reversed_dict.keys()
    keys.sort(reverse=True)
    big_list = list()
    for k in keys:
        big_list+=reversed_dict[k]
    return deduplicate_list_in_order(big_list)


def typification_exp1_single(vec_files, partition_file, DB2Vec_file, prefix_file, output_file):
    """
    Designed for single partition/output file
    :param vec_files:
    :param partition_file:
    :param DB2Vec_file:
    :param prefix_file:
    :param output_file:
    :return:
    """
    norm_vecs = _sum_and_normalize_vecs(vec_files)
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    model = Word2Vec.load(DB2Vec_file)
    model.init_sims(replace=True)
    print 'finished reading embeddings file'
    labels = list()
    instances = list()
    instance_vectors = list()
    with codecs.open(partition_file, 'r', 'utf-8') as f:
        for line in f:
            try:
                parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
                label = parsed_triple['object'].n3()[1:-1]
                instance = parsed_triple['subject'].n3()[1:-1]
                orig_instance = str(instance)
                for full_string, prefix in prefix_dict.items():
                    instance = instance.replace(full_string, prefix)
                labels.append(label)
                instances.append(orig_instance)
                instance_vectors.append(model[instance])
            except Exception as e:
                print e
                continue
    print 'finished reading in partition file...'
    del model
    print 'embedding model deleted...'
    out = codecs.open(output_file, 'w', 'utf-8')
    for i in range(len(instance_vectors)):
        answer = dict()
        score_dict = _compute_cosine_dict(instance_vectors[i], norm_vecs)
        answer[instances[i]] = score_dict
        json.dump(answer, out)
        out.write('\n')
    out.close()

def typification_exp1_full_run(partition_path, model, prefix_file):
    """
    You may want to optimize a little more for the larger partitions. For the small ones, we can afford
    to load a lot more in memory.
    :param partition_path:
    :param DB2Vec_file:
    :return:
    """
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    # model = Word2Vec.load(DB2Vec_file)
    # model.init_sims(replace=True)
    # print 'finished reading embeddings file'
    # output_files = list()

    for i in range(1, 6):
        output_file = partition_path+'exp1-small-partition-'+str(i)+'-score-dict.jl'
        vec_files = list()
        partition_file = partition_path+'small-partition-'+str(i)+'.ttl'
        for j in range(1,6):
            if j == i:
                continue
            else:
                vec_files.append(partition_path+'classVecsSimpleSum'+str(j)+'.jl')
        norm_vecs = _sum_and_normalize_vecs(vec_files)
        labels = list()
        instances = list()
        instance_vectors = list()
        with codecs.open(partition_file, 'r', 'utf-8') as f:
            for line in f:
                try:
                    parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
                    label = parsed_triple['object'].n3()[1:-1]
                    instance = parsed_triple['subject'].n3()[1:-1]
                    orig_instance = str(instance)
                    for full_string, prefix in prefix_dict.items():
                        instance = instance.replace(full_string, prefix)
                    labels.append(label)
                    instances.append(orig_instance)
                    instance_vectors.append(model[instance])
                except Exception as e:
                    print e
                    continue
        print 'finished reading in partition file...'
        # del model
        # print 'embedding model deleted...'
        out = codecs.open(output_file, 'w', 'utf-8')
        for i in range(len(instance_vectors)):
            answer = dict()
            score_dict = _compute_cosine_dict(instance_vectors[i], norm_vecs)
            answer[instances[i]] = score_dict
            json.dump(answer, out)
            out.write('\n')
        out.close()


def typification_exp2_full_run(partition_path, model, prefix_file):
    """
    You may want to optimize a little more for the larger partitions. For the small ones, we can afford
    to load a lot more in memory.
    :param partition_path:
    :param DB2Vec_file:
    :return:
    """
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]

    # output_files = list()

    for i in range(1, 6):
        output_file = partition_path+'exp2-small-partition-'+str(i)+'-score-dict.jl'
        vec_files = list()
        partition_file = partition_path + 'small-partition-' + str(i) + '-random.ttl'
        for j in range(1,6):
            if j == i:
                continue
            else:
                vec_files.append(partition_path+'classVecsCFSum'+str(j)+'.jl')
        norm_vecs = _sum_and_normalize_vecs(vec_files)
        labels = list()
        instances = list()
        instance_vectors = list()
        with codecs.open(partition_file, 'r', 'utf-8') as f:
            for line in f:
                try:
                    parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
                    label = parsed_triple['object'].n3()[1:-1]
                    instance = parsed_triple['subject'].n3()[1:-1]
                    orig_instance = str(instance)
                    for full_string, prefix in prefix_dict.items():
                        instance = instance.replace(full_string, prefix)
                    labels.append(label)
                    instances.append(orig_instance)
                    instance_vectors.append(model[instance])
                except Exception as e:
                    print e
                    continue
        print 'finished reading in partition file...'
        # del model
        # print 'embedding model deleted...'
        out = codecs.open(output_file, 'w', 'utf-8')
        for i in range(len(instance_vectors)):
            answer = dict()
            score_dict = _compute_cosine_dict(instance_vectors[i], norm_vecs)
            answer[instances[i]] = score_dict
            json.dump(answer, out)
            out.write('\n')
        out.close()


def typification_exp2_big_partition(partition_path, model, prefix_file, index):
    """
    You may want to optimize a little more for the larger partitions. For the small ones, we can afford
    to load a lot more in memory.
    :param partition_path:
    :param DB2Vec_file:
    :return:
    """
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]

    # output_files = list()
    vec_files = list()
    output_file = partition_path + 'exp2-partition-'+str(index)+'-score-dict.jl'
    for j in range(1,6):
        if j != index:
            vec_files.append(partition_path+'classVecsCFSum'+str(j)+'.jl')
    norm_vecs = _sum_and_normalize_vecs(vec_files)
    labels = list()
    instances = list()
    instance_vectors = list()
    partition_file = partition_path+'partition-'+str(index)+'.ttl'
    with codecs.open(partition_file, 'r', 'utf-8') as f:
        for line in f:
            try:
                parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
                label = parsed_triple['object'].n3()[1:-1]
                instance = parsed_triple['subject'].n3()[1:-1]
                orig_instance = str(instance)
                for full_string, prefix in prefix_dict.items():
                    instance = instance.replace(full_string, prefix)
                labels.append(label)
                instances.append(orig_instance)
                instance_vectors.append(model[instance])
            except Exception as e:
                print e
                continue
    print 'finished reading in partition file...'
    # del model
    # print 'embedding model deleted...'
    out = codecs.open(output_file, 'w', 'utf-8')
    for i in range(len(instance_vectors)):
        answer = dict()
        score_dict = _compute_cosine_dict(instance_vectors[i], norm_vecs)
        answer[instances[i]] = score_dict
        json.dump(answer, out)
        out.write('\n')
    out.close()


def typification_exp3_full_run(partition_path, model, prefix_file, type_statistics_file):
    """
    You may want to optimize a little more for the larger partitions. For the small ones, we can afford
    to load a lot more in memory.
    :param partition_path:
    :param DB2Vec_file:
    :return:
    """
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    type_weight_dict = build_type_weight_dict(type_statistics_file)
    # output_files = list()

    for i in range(1, 6):
        output_file = partition_path+'exp3-small-partition-'+str(i)+'-score-dict.jl'
        vec_files = list()
        partition_file = partition_path + 'small-partition-' + str(i) + '.ttl'
        for j in range(1,6):
            if j == i:
                continue
            else:
                vec_files.append(partition_path+'classVecsCFSum'+str(j)+'.jl')
        norm_vecs = _sum_and_normalize_vecs(vec_files)
        labels = list()
        instances = list()
        instance_vectors = list()
        with codecs.open(partition_file, 'r', 'utf-8') as f:
            for line in f:
                try:
                    parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
                    label = parsed_triple['object'].n3()[1:-1]
                    instance = parsed_triple['subject'].n3()[1:-1]
                    orig_instance = str(instance)
                    for full_string, prefix in prefix_dict.items():
                        instance = instance.replace(full_string, prefix)
                    labels.append(label)
                    instances.append(orig_instance)
                    instance_vectors.append(model[instance])
                except Exception as e:
                    print e
                    continue
        print 'finished reading in partition file...'
        # del model
        # print 'embedding model deleted...'
        out = codecs.open(output_file, 'w', 'utf-8')
        for i in range(len(instance_vectors)):
            answer = dict()
            score_dict = _compute_cosine_dict(instance_vectors[i], norm_vecs, mult_factors=type_weight_dict)
            answer[instances[i]] = score_dict
            json.dump(answer, out)
            out.write('\n')
        out.close()


def typification_exp4_zero_shot_single(vec_files, partition_file, DB2Vec_file, prefix_file, output_file):
    norm_vecs = _sum_and_normalize_vecs(vec_files)

    training_vecs = dict()
    test_vec = dict()
    types = norm_vecs.keys()
    random.shuffle(types)
    num_train = 5
    for i in range(0, num_train):
        training_vecs[types[i]] = norm_vecs[types[i]]
    test_vec[types[num_train]] = norm_vecs[types[num_train]]
    print 'training classes are...'
    print training_vecs.keys()
    print 'test class is...'
    print test_vec.keys()
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    model = Word2Vec.load(DB2Vec_file)
    model.init_sims(replace=True)
    print 'finished reading embeddings file'
    labels = list()
    instances = list()
    instance_vectors = list()
    with codecs.open(partition_file, 'r', 'utf-8') as f:
        for line in f:
            try:
                parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
                label = parsed_triple['object'].n3()[1:-1]
                if label not in training_vecs and label not in test_vec:
                    continue
                instance = parsed_triple['subject'].n3()[1:-1]
                orig_instance = str(instance)
                for full_string, prefix in prefix_dict.items():
                    instance = instance.replace(full_string, prefix)
                labels.append(label)
                instances.append(orig_instance)
                instance_vectors.append(model[instance])
            except Exception as e:
                print e
                continue
    print 'finished reading in partition file...'
    del model
    print 'embedding model deleted...'
    out = codecs.open(output_file, 'w', 'utf-8')
    for i in range(len(instance_vectors)):
        answer = dict()
        score_dict = _compute_cosine_dict(instance_vectors[i], training_vecs)
        answer[instances[i]] = score_dict
        answer['label'] = labels[i]
        json.dump(answer, out)
        out.write('\n')
    out.close()


def typification_exp4_zero_shot_alternative(partition_file, score_dict_file, output_file):
    """
    This one doesn't require reading in the embedding model, under the assumption we already have
    a score dict file and are using the scores previously generated.
    :param vec_files:
    :param score_dict_file:
    :param output_file:
    :return:
    """
    # norm_vecs = _sum_and_normalize_vecs(vec_files)

    # training_vecs = list()
    test_vec = list()

    # prefix_dict = dict()
    # with codecs.open(prefix_file, 'r', 'utf-8') as f:
    #     for line in f:
    #         fields = re.split('\t', line[0:-1])
    #         prefix_dict[fields[0]] = fields[1]
    # model = Word2Vec.load(DB2Vec_file)
    # model.init_sims(replace=True)
    # print 'finished reading embeddings file'
    # labels = list()
    instances = dict()
    label_set = set()
    # instance_vectors = list()
    with codecs.open(partition_file, 'r', 'utf-8') as f:
        for line in f:
            try:
                parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
                label = parsed_triple['object'].n3()[1:-1]
                # if label not in training_vecs and label not in test_vec:
                #     continue
                instance = parsed_triple['subject'].n3()[1:-1]
                orig_instance = str(instance)
                # for full_string, prefix in prefix_dict.items():
                #     instance = instance.replace(full_string, prefix)
                # labels.append(label)
                instances[orig_instance] = label # at this point, we assume one label per instance
                label_set.add(label)
                # instance_vectors.append(model[instance])
            except Exception as e:
                # print e
                continue
    print 'finished reading in partition file...'

    types = list(label_set)
    random.shuffle(types)
    num_train = 50
    num_test = 10
    training_vecs = types[0:num_train]
    test_vecs = types[num_train:num_train+num_test]
    print 'training classes are...'
    print training_vecs
    print 'test class is...'
    print test_vecs

    # del model
    # print 'embedding model deleted...'
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(score_dict_file, 'r', 'utf-8') as f:
        for line in f:
            try:
                answer = json.loads(line)
                key = answer.keys()[0]
                if key not in instances:
                    continue
                elif instances[key] not in training_vecs and instances[key] not in test_vecs:
                    continue
                # answer = dict()
                # score_dict = _compute_cosine_dict(instance_vectors[i], training_vecs)
                new_dict = dict()
                for t in training_vecs:
                    new_dict[t] = answer[key][t]
                answer[key] = new_dict
                answer['label'] = instances[key]
                json.dump(answer, out)
                out.write('\n')
            except Exception as e:
                print e
                continue
    out.close()


def _compute_cosine_dict(vector, dict_of_vectors, mult_factors=None):
    """

    :param vector:
    :param dict_of_vectors:
    :return: a dict of scores
    """
    answer = dict()
    for k, v in dict_of_vectors.items():
        if mult_factors:
            if k in mult_factors:
                answer[k] = abs_cosine_sim(vector, v)*mult_factors[k]
                # print answer[k]
            else:
                print k,' not in mult factors...'
                continue
        else:
            answer[k] = abs_cosine_sim(vector, v)
    return answer

def _sum_and_normalize_vecs(vec_files):
    """
    Can be used for any set of vector files
    :param vec_files: a list of vec jl files e.g. classVecsSimpleSum
    :return: a dictionary of normalized vectors
    """
    norm_vecs = dict()
    for vec_file in vec_files:
        with codecs.open(vec_file, 'r', 'utf-8') as f:
            for line in f:
                vec = json.loads(line)
                for k, v in vec.items():
                    if k not in norm_vecs:
                        norm_vecs[k] = np.array(v)
                    else:
                        norm_vecs[k] = np.sum([norm_vecs[k], np.array(v)], axis=0)
    for k, v in norm_vecs.items():
        norm_vecs[k] = l2_normalize(v)
    print 'returning ',len(norm_vecs),' normalized vectors...'
    return norm_vecs

def sum_normalize_all_vecs(vec_files, output_file):
    norm_vecs = _sum_and_normalize_vecs(vec_files)
    # for k in norm_vecs.keys():
    #     norm_vecs[k] = norm_vecs[k].tolist()
    out = codecs.open(output_file, 'w', 'utf-8')
    json.dump(norm_vecs, out)
    out.close()



def build_instance_type_dict(partition_files):
    """
    for each instance, record its classes as a deduplicated list and return the dictionary
    :param partition_files:
    :return: a dict of instance-class list
    """
    answer = dict()
    types = set()
    for partition_file in partition_files:
        print 'processing partition file...',partition_file
        with codecs.open(partition_file, 'r', 'utf-8') as f:
            for line in f:
                parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
                subject_uri = parsed_triple['subject'].n3()[1:-1]
                object_uri = parsed_triple['object'].n3()[1:-1]
                if subject_uri not in answer:
                    answer[subject_uri] = list()
                answer[subject_uri].append(object_uri)
                types.add(object_uri)
    for k in answer.keys():
        answer[k] = list(set(answer[k]))
    print 'num types: ',len(types)
    return answer

def build_type_weight_dict(type_statistics_file):
    type_weight_dict = dict()
    types = list()
    scores = list()
    with codecs.open(type_statistics_file, 'r', 'utf-8') as f:
        for line in f:
            fields = line[0:-1].split('\t')
            types.append(fields[0])
            scores.append(fields[1])
    norm_scores = l2_normalize(scores)
    for i in range(len(norm_scores)):
        type_weight_dict[types[i]] = norm_scores[i]
    return type_weight_dict


def build_subclass_dict_from_db_onto(pruned_ontology, type_statistics_file, output_file):
    types = build_type_weight_dict(type_statistics_file).keys()
    subclass_dict = dict() # remember, key is the superclass
    with codecs.open(pruned_ontology, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            subject_uri = parsed_triple['subject'].n3()[1:-1]
            object_uri = parsed_triple['object'].n3()[1:-1]

            if subject_uri not in types or object_uri not in types \
                    or parsed_triple['predicate'] != URIRef(u'http://www.w3.org/2000/01/rdf-schema#subClassOf'):
                continue
            else:
                if object_uri not in subclass_dict:
                    subclass_dict[object_uri] = set()
                subclass_dict[object_uri].add(subject_uri)
    for k in subclass_dict.keys():
        subclass_dict[k] = list(subclass_dict[k])
        subclass_dict[k].sort()

    out =codecs.open(output_file, 'w', 'utf-8')
    json.dump(subclass_dict, out)
    out.close()

def build_unpruned_subclass_dict_from_db_onto(pruned_ontology, output_file):
    # types = build_type_weight_dict(type_statistics_file).keys()
    subclass_dict = dict() # remember, key is the superclass
    with codecs.open(pruned_ontology, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            subject_uri = parsed_triple['subject'].n3()[1:-1]
            object_uri = parsed_triple['object'].n3()[1:-1]

            if parsed_triple['predicate'] != URIRef(u'http://www.w3.org/2000/01/rdf-schema#subClassOf'):
                continue
            else:
                if object_uri not in subclass_dict:
                    subclass_dict[object_uri] = set()
                subclass_dict[object_uri].add(subject_uri)
    for k in subclass_dict.keys():
        subclass_dict[k] = list(subclass_dict[k])
        subclass_dict[k].sort()

    out = codecs.open(output_file, 'w', 'utf-8')
    json.dump(subclass_dict, out)
    out.close()


def convert_exp1_instance_format_to_exp1_class_format(instance_format_file, partition_files, output_file, topk=None):
    """
    We use a simple score-majority method. Each class (per instance) is assigned the weight
    of that instance divided by the number of classes for that instance.
    :param instance_format_file:
    :param class_format_file:
    :param output_file:
    :return:
    """
    instance_type_dict = build_instance_type_dict(partition_files)
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(instance_format_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            new_dict = dict()
            score_dict = obj.values()[0]
            if topk:
                score_dict = extract_top_k(reverse_dict(obj.values()[0]), topk)
            for k in score_dict.keys():
                if k in instance_type_dict:
                    types = instance_type_dict[k]
                    for typ in types:
                        if typ not in new_dict:
                            new_dict[typ] = 0.0
                        new_dict[typ] += obj.values()[0][k]*1.0/len(types)
            obj[obj.keys()[0]] = new_dict
            json.dump(obj, out)
            out.write('\n')
    out.close()


def convert_baseline_file_to_exp1_instance_format(baseline_file, prefix_file, output_file):
    """
    This is not the only preprocessing that needs to be done. We also need to take the output file
    and deduce classes.
    :param baseline_file:
    :param prefix_file:
    :param output_file:
    :return:
    """
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(baseline_file, 'r', 'utf-8') as f:
        for line in f:
            fields = line[0:-1].split('\t')
            score_dict = dict()
            for i in range(1, len(fields), 2):
                for full_string, prefix in prefix_dict.items():
                    fields[i] = fields[i].replace(prefix, full_string)
                score_dict[fields[i]] = float(fields[i+1])
            for full_string, prefix in prefix_dict.items():
                fields[0] = fields[0].replace(prefix, full_string)
            answer = dict()
            answer[fields[0]] = score_dict
            json.dump(answer, out)
            out.write('\n')

    out.close()


def prune_dbpedia_instances_file(instances_file, DB2Vec_file, prefix_file, output_file):
    """
    We need to prune the instances file so that both subject and object actually exist in the DB2Vec file.
    :param instances_file:
    :param DB2Vec_file:
    :param prefix_file:
    :param output_file:
    :return:
    """
    prefix_dict = dict()
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    model = Word2Vec.load(DB2Vec_file)
    model.init_sims(replace=True)
    print 'finished reading embeddings file'
    count = 0
    found = 0
    not_found = 0
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(instances_file, 'r', 'utf-8') as f:
        for line in f:
            # print 'processing line...',
            # print count
            count += 1
            if count % 10000==0:
                print 'processing line...',count
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            if not parsed_triple:
                print 'line is not triple...',
                print line
                continue
            if not parsed_triple['isObjectURI']:
                print 'object is not a URI...',
                print line
                continue
            # if parsed_triple['predicate'] != URIRef(u'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
            #     print 'predicate is not type...',
            #     print line
            #     continue
            subject_uri = parsed_triple['subject'].n3()[1:-1]
            object_uri = parsed_triple['object'].n3()[1:-1]
            for full_string, prefix in prefix_dict.items():
                subject_uri = subject_uri.replace(full_string, prefix)
                object_uri = object_uri.replace(full_string, prefix)
            try:
                subject_embedding_vector = model[subject_uri]
                object_embedding_vector = model[object_uri]
                found += 1
                out.write(line)
            except KeyError:
                # print 'uri not found in embeddings:',
                # print uri
                not_found += 1
                continue
    out.close()
    print 'input file processing complete. lines written: ',
    print found
    print 'lines not written: ',
    print not_found


def _is_class_in_DB2Vec(DB2Vec_file, prefix_file, type_file):
    prefix_dict = dict()
    result = False
    with codecs.open(prefix_file, 'r', 'utf-8') as f:
        for line in f:
            fields = re.split('\t', line[0:-1])
            prefix_dict[fields[0]] = fields[1]
    model = Word2Vec.load(DB2Vec_file)
    model.init_sims(replace=True)
    print 'finished reading embeddings file'
    count = 1
    with codecs.open(type_file, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            if count > 100: break
            count += 1
            object_uri = parsed_triple['object'].n3()[1:-1]
            for full_string, prefix in prefix_dict.items():
                object_uri = object_uri.replace(full_string, prefix)
            try:

                object_embedding_vector = model[object_uri]
                result = True
                print line
                break
            except KeyError:
                # print 'uri not found in embeddings:',
                # print uri

                continue
    print result


def compute_instance_class_frequencies(types_file, output_file):
    """
    A relatively simple program that computes the number of types corresponding to an instance
    :param types_file:
    :param output_file:
    :return:
    """
    inst_type_dict = build_instance_type_dict([types_file])
    out = codecs.open(output_file, 'w')
    for k, v in inst_type_dict.items():
        out.write(k+'\t'+str(len(v))+'\n')
    out.close()

def obtain_pruned_non_typed_instances(pruned_instances_file, pruned_types_file, output_file):
    """
    We didn't find any untyped instances in the pruned instances file, it turns out.
    :param pruned_instances_file:
    :param pruned_types_file:
    :param output_file:
    :return:
    """
    typed_instances = set()
    with codecs.open(pruned_types_file, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            subject_uri = parsed_triple['subject'].n3()[1:-1]
            typed_instances.add(subject_uri)
    print 'finished reading pruned_types_file...'
    out = codecs.open(output_file, 'w', 'utf-8')
    typed = 0
    untyped = 0
    with codecs.open(pruned_instances_file, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            subject_uri = parsed_triple['subject'].n3()[1:-1]
            if subject_uri not in typed_instances:
                out.write(line)
                untyped += 1
            else:
                typed += 1
    out.close()
    print 'finished processing. Number of untyped-instance triples found is...',str(untyped)
    print 'Number of typed-instance triples found is...',str(typed)


def remove_literal_triples_from_dbpedia_ontology(dbpedia_ontology, output_file):
    """
    remove all triples from the ontology that have a literal for an object
    :param dbpedia_ontology:
    :param output_file:
    :return:
    """
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(dbpedia_ontology, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            if not parsed_triple:
                # print 'line is not triple...',
                # print line
                continue
            if not parsed_triple['isObjectURI']:
                # print 'object is not a URI...',
                # print line
                continue
            out.write(line)
    out.close()

def superclass_tsne_visualization(classVecs_file, pruned_subclass_file, output_file):
    classVecs_dict = json.load(codecs.open(classVecs_file, 'r'))
    subclass_dict = json.load(codecs.open(pruned_subclass_file, 'r'))
    # supertypes = ['http://dbpedia.org/ontology/Animal', 'http://dbpedia.org/ontology/Politician',
    #          'http://dbpedia.org/ontology/Building', 'http://dbpedia.org/ontology/MusicalWork',
    #          'http://dbpedia.org/ontology/Plant'] # let's not leave this to chance
    supertypes = ['http://dbpedia.org/ontology/SportsTeam', 'http://dbpedia.org/ontology/SportsLeague',
             'http://dbpedia.org/ontology/Company', 'http://dbpedia.org/ontology/Organisation',
             'http://dbpedia.org/ontology/EducationalInstitution']
    y_labels = list()
    y_superclasses = list()
    y_subclasses = list()
    x = list()
    count = 1
    for supertype in supertypes:
        for subtype in subclass_dict[supertype]:
            x.append(classVecs_dict[subtype])
            y_labels.append(count)
            y_superclasses.append(supertype)
            y_subclasses.append(subtype)
        count += 1
    Y = tsne.tsne(np.array(x), 2, 50, 10.0)
    scatter(Y[:, 0], Y[:, 1], 10, y_labels)
    print 'Total number of items are : ',
    print len(y_labels)
    out = codecs.open(output_file, 'w', 'utf-8')
    out.write('x_label,y_label,superclass,subclass,label\n')
    for i in range(0, len(y_labels)):
        out.write(str(Y[i,0]) + ',' + str(Y[i,1]) + ','+y_superclasses[i]+ ','+y_subclasses[i]+','
                  +str(y_labels[i]))
        out.write('\n')
    out.close()

    show()

def thread_trial(k, d):
    time.sleep(k)

def process_samples_for_labeling(sample_file, output_file):
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(sample_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                m = obtain_ranked_list_from_score_dict(v)
                if len(m) > 10:
                    obj[k] = m[0:10]
                else:
                    obj[k] = m
            json.dump(obj, out)
            out.write('\n')
    out.close()

def pad_sample_file(sample_file, type_statistics_file, output_file):
    types = build_type_weight_dict(type_statistics_file).keys()
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(sample_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                if len(v) == 10:
                    json.dump(obj, out)
                    out.write('\n')
                    break
                pad_types = list((set(types)).difference(set(v)))
                random.shuffle(pad_types)
                for pad_type in pad_types:
                    v.append(pad_type)
                    if len(v) == 10:
                        break
                json.dump(obj, out)
                out.write('\n')

    out.close()


def compute_unique_object_subject_counts(triples_file):
    instances = set()
    types = set()
    with codecs.open(triples_file, 'r', 'utf-8') as f:
        for line in f:
            parsed_triple = EmbeddingGenerator.EmbeddingGenerator.parse_line_into_triple(line)
            types.add(parsed_triple['object'])
            instances.add(parsed_triple['subject'])
    print 'num unique instances: ',len(instances)
    print 'num unique types :',len(types)

def count_num_results_in_baselines(baseline_file):
    # exp1_dict = dict()
    count = 0
    total = 0
    with codecs.open(baseline_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            total += 1
            for k, v in obj.items():
                count += len(v.keys())
    # print 'number of types with non-zero probability scores per entity: ',str(count*1.0/total)
    print str(count * 1.0 / total)

def count_better_scores(topical_relevance_file):
    count = 0
    with codecs.open(topical_relevance_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            count += obj['better_score']
    print count

def topical_relevance_statistics(topical_relevance_file):
    relevance = list()
    with codecs.open(topical_relevance_file, 'r', 'utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception as e:
                print e
                print line
            relevance.append(obj['relevant_score']*1.0/10)
    print np.mean(relevance)
    print np.std(relevance)


# path = '/Users/mayankkejriwal/datasets/eswc2017/'
# partition_path = path+'eswc2017/dbpedia-experiments/types-partitions/topical-relevance-results/'
# topical_relevance_statistics(partition_path+'xx-sp1-100-random-padded.jl')
# topical_relevance_statistics(partition_path+'ex-sp1-100-random.jl')
# compute_unique_object_subject_counts(partition_path+'partition-5.ttl')
# pad_sample_file(partition_path+'xx-sp1-100-random.jl', partition_path+'pruned-types-statistics.txt',
#                 partition_path+'xx-sp1-100-random-padded.jl')
# process_samples_for_labeling(partition_path+'baseline-class-1-10nn-random-100.jl',
#                              partition_path+'xx-sp1-100-random.jl')
# superclass_tsne_visualization(partition_path+'classVecsCFSum.json', partition_path+'pruned-subclass-dict.json',
#                           partition_path+'types-partitions/tsne-visualizations/coherent.csv'    )
# sum_normalize_all_vecs([partition_path+'classVecsCFSum1.jl',partition_path+'classVecsCFSum2.jl',
#     partition_path + 'classVecsCFSum3.jl', partition_path+'classVecsCFSum4.jl',
#                                 partition_path+'classVecsCFSum5.jl'],partition_path+'classVecsCFSum.json')
# build_subclass_dict_from_db_onto(path+'eswc2017/curated-dbpedia-ontology.nt',
#                             partition_path+'pruned-types-statistics.txt', partition_path+'pruned-subclass-dict.json')
# build_unpruned_subclass_dict_from_db_onto(path+'dbpedia-files/curated-dbpedia-ontology.nt', path+'dbpedia-files/subclass_dict.json')
# exp_recall_analysis_big_partition(partition_path+'exp2-partition-1-score-dict.jl',
#                 partition_path+'partition-1.ttl', partition_path+'recall@k/exp2-big-partition-1.csv')
# exp_recall_analysis(partition_path+'baseline-class-5-10nn.jl',
#                 partition_path+'small-partition-5-random.ttl', partition_path+'recall@k/nn10-small-partition-5.csv')
# remove_literal_triples_from_dbpedia_ontology(path+'eswc2017/dbpedia_2016-04.nt', path+'eswc2017/curated-dbpedia-ontology.nt')
# obtain_pruned_non_typed_instances(partition_path+'pruned-instances.ttl', partition_path+'pruned-types.ttl',
#                                   partition_path + 'pruned-untyped-instances.ttl')
# time.sleep(12000) # in seconds
# get_random_subset_of_files([partition_path+'exp2-small-partition-1-score-dict.jl', partition_path+'baseline-class-1-10nn.jl'],
#     [partition_path+'exp2-small-partition-1-score-dict-random-100.jl',partition_path+'baseline-class-1-10nn-random-100.jl'])
# _is_class_in_DB2Vec(path + 'heiko-vectors/DB2Vec_sg_500_5_5_15_4_500',path + 'heiko-vectors/prefix.txt',partition_path+'pruned-types.ttl')
# dbpedia_typification_baseline([partition_path+'small-partition-1-random.ttl',partition_path+'small-partition-2-random.ttl',
# partition_path+'small-partition-3-random.ttl',partition_path+'small-partition-4-random.ttl',partition_path+'small-partition-5-random.ttl'],
#                               path + 'heiko-vectors/DB2Vec_sg_500_5_5_15_4_500',
#                               path + 'heiko-vectors/prefix.txt',
# [partition_path + 'baseline-1.tsv', partition_path + 'baseline-2.tsv',partition_path + 'baseline-3.tsv',
#  partition_path + 'baseline-4.tsv', partition_path + 'baseline-5.tsv'])
# typification_exp1_single([partition_path+'classVecsSimpleSum1.jl',partition_path+'classVecsSimpleSum2.jl',
#                    partition_path+'classVecsSimpleSum4.jl',partition_path+'classVecsSimpleSum5.jl',],
# partition_path+'small-partition-3.ttl',path + 'heiko-vectors/DB2Vec_sg_500_5_5_15_4_500',
#                   path + 'heiko-vectors/prefix.txt', partition_path+'exp1-small-partition-3-score-dict.jl')
# DB2Vec_file = path + 'heiko-vectors/DB2Vec_sg_500_5_5_15_4_500'
# model = Word2Vec.load(DB2Vec_file)
# model.init_sims(replace=True)
# print 'finished reading embeddings file'
# typification_exp2_big_partition(partition_path, model,
#                            path + 'heiko-vectors/prefix.txt', index=4)
# typification_exp2_big_partition(partition_path, model,
#                            path + 'heiko-vectors/prefix.txt', index=5)
# # print build_type_weight_dict(path+'eswc2017/dbpedia-experiments/pruned-types-statistics.txt')

# typification_exp3_full_run(partition_path, model,
#                            path + 'heiko-vectors/prefix.txt',path+'eswc2017/dbpedia-experiments/pruned-types-statistics.txt')
# typification_exp2_full_run(partition_path, model,
#                            path + 'heiko-vectors/prefix.txt')

# convert_baseline_file_to_exp1_instance_format(partition_path+'baseline-5.tsv',path + 'heiko-vectors/prefix.txt',
#                                      partition_path+'baseline-5.jl')
# convert_exp1_instance_format_to_exp1_class_format(partition_path+'baseline-5.jl',
#         [partition_path + 'partition-3.ttl',partition_path + 'partition-2.ttl',partition_path + 'partition-4.ttl',
#          partition_path + 'partition-1.ttl'],
# d= build_instance_type_dict([partition_path + 'partition-3.ttl',partition_path + 'partition-2.ttl',partition_path + 'partition-4.ttl',
#          partition_path + 'partition-1.ttl', partition_path + 'partition-5.ttl'])
# print 'num instances: ',
# print len(set(d.keys()))
# print 'num types: ',
# print len(set(d.values()))
#                                 partition_path + 'baseline-class-5-5nn.jl', topk=5)
# typification_exp4_zero_shot_alternative(partition_path+'small-partition-1-random.ttl',partition_path+'exp2-small-partition-1-score-dict.jl',
#                                         partition_path+'zs-small-partition-1.jl')
# zero_shot_analysis_binary(partition_path+'zs-small-partition-1.jl')
# compute_instance_class_frequencies(partition_path+'pruned-types.ttl', partition_path+'instance-cf.tsv')
# for i in range(1, 6):
#     count_num_results_in_baselines(partition_path+'baseline-class-'+str(i)+'-10nn.jl')
#     exp_recall_analysis(partition_path+'baseline-class-'+str(i)+'-5nn.jl',
#                  partition_path+'small-partition-'+str(i)+'-random.ttl', partition_path+'recall@k/nn5-small-partition-'+str(i)+'.csv')
# random_baseline_analysis(partition_path+'exp2-small-partition-2-score-dict.jl', partition_path+'small-partition-2.ttl', False)
# exp_analysis(partition_path+'baseline-class-1-10nn.jl', partition_path+'small-partition-1-random.ttl', False)
# m = 'dbr:Dr\xc3\xa9an'
# out = codecs.open(path + 'eswc2017/dbpedia-experiments/types-partitions/tmp', 'w')
# out.write(m)
# out.close()
# print m
# prune_dbpedia_types_file(path+'eswc2017/instance_types_en.ttl', path+'heiko-vectors/DB2Vec_sg_500_5_5_15_4_500',
#                          path+'heiko-vectors/prefix.txt', path+'eswc2017/dbpedia-experiments/pruned-types.ttl')
# prune_dbpedia_instances_file(path+'eswc2017/dbpedia_mappingbased_objects_en.ttl', path+'heiko-vectors/DB2Vec_sg_500_5_5_15_4_500',
#                          path+'heiko-vectors/prefix.txt', path+'eswc2017/dbpedia-experiments/pruned-instances.ttl')
# partition_file(path+'eswc2017/dbpedia-experiments/pruned-instances.ttl',
#                          path+'eswc2017/dbpedia-experiments/instances-partitions/')
# dbpedia_types_statistics(path+'eswc2017/dbpedia-experiments/pruned-types.ttl',
#                          path+'eswc2017/dbpedia-experiments/pruned-types-statistics.txt')
