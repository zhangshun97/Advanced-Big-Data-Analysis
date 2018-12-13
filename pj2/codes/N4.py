from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("N4_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)

# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt") \
    .map(lambda x: x.split('\t'))\
    .map(lambda x: ((x[11], x[3]), 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda x: (x[0][0], (x[0][1], x[1]))) \
    .groupByKey() \
    .takeOrdered(10, lambda x: -sum([a[1] for a in x[1]]))

with open('N4_15300180012_output.txt', 'w') as f:
    f.write('N4_15300180012_output\n')

for city, iterative in data:
    data_ = sc.parallelize(iterative).takeOrdered(3, lambda x: -x[1])
    with open('N4_15300180012_output.txt', 'a') as f:
        f.write('----------\n')
        f.write(city + '\n')
        for name, num in data_:
            f.write(name)
            f.write('   ')
            f.write(str(num))
            f.write('\n')

sc.stop()
