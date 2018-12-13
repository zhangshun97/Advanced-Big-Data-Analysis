from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("H_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)


# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt")\
    .map(lambda x: x.split('\t'))\
    .filter(lambda x: x[6] == 'K')\
    .map(lambda x: (x[2], 1))\
    .reduceByKey(lambda a, b: a + b)\
    .takeOrdered(5, lambda x: -x[1])

print data

sc.stop()
