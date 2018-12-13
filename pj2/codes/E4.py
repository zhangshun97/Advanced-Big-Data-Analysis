from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("E4_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)


def get_birth_month(record):
    "for E4, to get birth month from birth date"
    record = record.split('\t')
    birth_date_str = record[8]
    birth = birth_date_str.split('/')
    if birth[1] == '':
        return 'censored', 1
    else:
        return birth[1], 1


# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt")\
    .map(get_birth_month)\
    .reduceByKey(lambda a, b: a + b)\
    .collect()


print data

sc.stop()
