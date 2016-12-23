import codecs
import json
from rdflib import Graph
from rdflib.term import Literal


def get_ordered_list_of_types(pruned_types_file):
    types = list()
    with codecs.open(pruned_types_file, 'r', 'utf-8') as f:
        for line in f:
            types.append(line[0:-1].split('\t')[0])
    return types


def compactify_fuzzy_cluster_file(fuzzy_cluster_file, pruned_types_file, output_file):
    """
    Each fuzzy cluster file is about 20 GB which is way too big to process on a laptop. I'm going to
    use the pruned_types_dict to significantly reduce the imprint without losing any information.
    Each key will now reference a list of floats representing independently
    computed probabilities (hence, this is not really a distribution from what
    I understand, but may have to re-check); the order is per the order in pruned_types_file.
    Note that if a type does not show up in an object, as it often doesn't, we give it a 0.0 score.
    It is almost always the case that we do not have 0 probabilities otherwise, so this gives
    us information about what types are missing also.
    :param fuzzy_cluster_file:
    :param pruned_types_file:
    :return:
    """
    types = get_ordered_list_of_types(pruned_types_file)
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(fuzzy_cluster_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            probs = obj.values()[0]
            list_probs = list()
            res = dict()
            for t in types:
                if t not in probs:
                    list_probs.append(0.0)
                else:
                    list_probs.append(round(probs[t],3))
            if len(list_probs) != 415:
                raise Exception
            res[obj.keys()[0]] = list_probs
            json.dump(res, out)
            out.write('\n')
    out.close()


def bfs_on_subclass_dict(subclass_dict_file, root_class, output_file):
    """
    The root class should ideally be a category class (e.g. Agent) that we're interested in. The subclass_dict
    should NOT be pruned, since there are broken branches in the pruned
    dict.
    :param subclass_dict_file:
    :param root_class:
    :param output_file:
    :return: None
    """
    subclass_dict = json.load(codecs.open(subclass_dict_file, 'r'))
    if root_class not in subclass_dict:
        raise Exception('Error! your root class is not in the provided ontology')
    list_to_explore = list()
    list_to_explore.append(root_class)
    class_set = set()
    while list_to_explore:
        cl = list_to_explore[0]
        class_set.add(cl)
        if len(list_to_explore) > 1:
            list_to_explore = list_to_explore[1:]
        else:
            list_to_explore = list()
        if cl in subclass_dict:
            list_to_explore += subclass_dict[cl]
    out = codecs.open(output_file, 'w', 'utf-8')
    obj = dict()
    list_class_set = list(class_set)
    list_class_set.sort()
    obj[root_class] = list_class_set
    json.dump(obj, out)
    out.close()


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
    for s, p, o in g:
        answer['subject'] = s
        answer['predicate'] = p
        answer['object'] = o

    if 'subject' not in answer:
        return None
    else:
        answer['isObjectURI'] = (type(answer['object']) != Literal)
        return answer

def subset_type_assertions_on_dict(types_file, root_class_dict_file, output_file):
    relevant_types = set(json.load(codecs.open(root_class_dict_file, 'r')).values()[0])
    print relevant_types
    out = codecs.open(output_file, 'w', 'utf-8')
    with codecs.open(types_file, 'r', 'utf-8') as f:
        for line in f:
            object = parse_line_into_triple(line)['object'].n3()[1:-1]
            if object in relevant_types:
                out.write(line)
    out.close()


def non_person_agent(agent_file, person_file, output_file):
    """
    Beware, the key in the output file is not a legal dbpedia type!
    :param agent_file:
    :param person_file:
    :param output_file:
    :return:
    """
    person_types = set(json.load(codecs.open(person_file, 'r')).values()[0])
    agent_types = set(json.load(codecs.open(agent_file, 'r')).values()[0])
    class_set = agent_types.difference(person_types)
    out = codecs.open(output_file, 'w', 'utf-8')
    obj = dict()
    list_class_set = list(class_set)
    list_class_set.sort()
    obj['NonPerson-Agent'] = list_class_set
    json.dump(obj, out)
    out.close()


# path = '/Users/mayankkejriwal/datasets/eswc2017/dbpedia-experiments/'
# inner_path = path+'types-partitions/zs-learning-data/'
# non_person_agent(inner_path+'agent_dict.json',inner_path+'person_dict.json',inner_path+'nonperson_agent_dict.json')
# subset_type_assertions_on_dict(path+'pruned-types.ttl', inner_path+'nonperson_agent_dict.json',
#                                inner_path+'nonperson_agent_instances.ttl')
# bfs_on_subclass_dict(inner_path+'subclass_dict.json', 'http://dbpedia.org/ontology/TopicalConcept',
#                      inner_path+'topical_concept_dict.json')
# compactify_fuzzy_cluster_file(inner_path+'exp2-partition-5-score-dict.jl',path+'pruned-types-statistics.txt',
#                               inner_path+'compactified-partition-5.jl')


