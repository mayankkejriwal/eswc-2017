import codecs
import re

class DBpediaSubsets:
    """
    The goal is to generate triples-files that are only a subset of the DBpedia triples. We write
    custom code for each benchmark that we use for this purpose
    """

    @staticmethod
    def generateDBpediaResourcesFile(input_file, output_file, resource_index):
        """

       :param input_file: Should be a tsv file with a header
       :param output_file: will be a simple text file with a dbpedia resource on each line. Ordering between
       input_file and this file does not exist; we do guarantee that a resource only occurs once in this file.
       :param resource_index: the column (a python-array index like 0 or -4) where the dbpedia resource
       in the input file may be found.
       :return:
        """
        header = True
        resources = set()
        out = codecs.open(output_file, 'w', 'utf-8')
        with codecs.open(input_file, 'r', 'utf-8') as f:
            for line in f:
                if header:
                    header = False
                    continue # ignore header
                resources.add(re.split('\t', line)[resource_index])

        for resource in resources:
            out.write(resource)
            out.write('\n')
        out.close()

    @staticmethod
    def combineDBpediaResources(LOD_folder='/Users/mayankkejriwal/datasets/eswc2017/LOD-ML-data/',
                    sub_folder_names=['aaup/', 'cities/', 'forbes/', 'metacriticAlbums/', 'metacriticMovies/'],
                    output_file_name='DBpediaResources_all.txt'):
        """
        We take the five sub-folders in the LOD_folder and the DBpediaResources.txt in each of them, and read
        it all into a set. Then we write the set out to an output file. This way we have a single point
        of reference that we can use to generate the DBpedia subset.
        """
        all_set = set()
        for sub_folder in sub_folder_names:
            with codecs.open(LOD_folder+sub_folder+'DBpediaResources.txt', 'r', 'utf-8') as f:
                for line in f:
                    all_set.add(line)

        out = codecs.open(LOD_folder+output_file_name, 'w', 'utf-8')
        for resource in all_set:
            out.write(resource)
        out.close()

    @staticmethod
    def getDBpediaTriplesSubset(resource_file='/Users/mayankkejriwal/datasets/eswc2017/LOD-ML-data/DBpediaResources_all.txt',
                                DBpedia_file='/Users/mayankkejriwal/datasets/eswc2017/dbpedia_mappingbased_objects_en.ttl',
                                output_file='/Users/mayankkejriwal/datasets/eswc2017/LOD-ML-data/DBpediaTriplesSubset.ttl'):
        """
        We will print out any resources for which we did not find even one triple in the DBpedia_file
        :param resource_file: A file with a DBpedia resource in each line
        :param DBpedia_file: This is the full file with triples
        :param output_file: A subset of the DBpedia file. The only requirement is that a resource from
        resource_file must be in the triple for that triple to get written out to output_file
        :return: None
        """
        # first let's read in all resources into a set
        resources = set()
        with codecs.open(resource_file, 'r', 'utf-8') as f:
            for line in f:
                resources.add(line[0:-1])
        out = codecs.open(output_file, 'w', 'utf-8')
        found_once = set()
        count = 1
        with codecs.open(DBpedia_file, 'r', 'utf-8') as f:
            for line in f:
                for resource in resources:
                    if resource in line:
                        out.write(line)
                        found_once.add(resource)
                print 'processed triple ',
                print count
                count += 1
        out.close()
        if resources.difference(found_once):
            print 'these resources have no triples...'
            print resources.difference(found_once)



# path = '/Users/mayankkejriwal/datasets/eswc2017/LOD-ML-data/'
# DBpediaSubsets.getDBpediaTriplesSubset()
# DBpediaSubsets.combineDBpediaResources()
# DBpediaSubsets.generateDBpediaResourcesFile(path+'completeDataset.tsv', path+'DBpediaResources.txt', 2)
