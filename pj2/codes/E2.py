from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("E2_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)


def get_name_letters(record):
    "for E2, to get the NAME letters"
    record = record.split('\t')
    name = record[2] + record[3]
    return name


# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt") \
    .map(get_name_letters) \
    .flatMap(lambda x: list(x)) \
    .map(lambda x: (x, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .takeOrdered(3, lambda x: -x[1])

print data

sc.stop()
