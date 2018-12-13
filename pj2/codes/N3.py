from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("N3_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)


def get_age(record):
    "for N3, to get age from birth date"
    birth_date_str = record[8]
    birth = birth_date_str.split('/')
    try:
        birth[2] = int(birth[2])
    except ValueError:
        birth[2] = 5000
    age = 2018 - birth[2]
    return age


# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt")\
    .map(lambda x: x.split('\t'))\
    .map(lambda x: (x[11], (get_age(x), 1)))\
    .filter(lambda x: x[1] >= 0)\
    .reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )\
    .map(lambda x: (x[0], 1.0*x[1][0]/x[1][1]))\
    .takeOrdered(5, lambda x: x[1])

print data

sc.stop()
