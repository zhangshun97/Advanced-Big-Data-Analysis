from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("E5_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)


# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt")\
    .map(lambda x: x.split('\t'))\
    .map(lambda record: (record[6], 1))\
    .reduceByKey(lambda a, b: a + b)\
    .collect()


print data

sc.stop()
