from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("N5_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)


def get_birth_month(record):
    "for E4, to get birth month from birth date"
    birth_date_str = record[8]
    birth = birth_date_str.split('/')
    if birth[1] == '':
        return 'censored'
    else:
        return birth[1]


# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt") \
    .map(lambda x: x.split('\t'))\
    .map(lambda x: ((x[11], get_birth_month(x)), 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda x: (x[0][0], (x[0][1], x[1]))) \
    .groupByKey() \
    .takeOrdered(10, lambda x: -sum([a[1] for a in x[1]]))

with open('N5_15300180012_output.txt', 'w') as f:
    f.write('N5_15300180012_output\n')

for city, iterative in data:
    data_ = sc.parallelize(iterative).takeOrdered(2, lambda x: -x[1])
    with open('N5_15300180012_output.txt', 'a') as f:
        f.write('----------\n')
        f.write(city + '\n')
        for name, num in data_:
            f.write(name)
            f.write('   ')
            f.write(str(num))
            f.write('\n')

sc.stop()
