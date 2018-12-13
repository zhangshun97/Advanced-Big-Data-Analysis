from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("E1_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)


def get_days(record):
    "for E1, to compare ages from birth date"
    record = record.split('\t')
    birth_date_str = record[8]
    birth = birth_date_str.split('/')
    for i in range(3):
        try:
            birth[i] = int(birth[i])
        except ValueError:
            if i == 0:
                birth[i] = 31
            elif i == 1:
                birth[i] = 12
            else:
                birth[i] = 2018
    days = (int(birth[2]) - 1800) * 1000 + int(birth[1]) * 50 + int(birth[0])
    return days


# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt")\
    .filter(lambda x: x.split('\t')[6] == 'E')\
    .takeOrdered(1, key=get_days)

print data

sc.stop()
