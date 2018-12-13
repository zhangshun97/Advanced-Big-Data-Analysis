from pyspark import SparkContext, SparkConf


# initialization SparkContext
conf = SparkConf().setAppName("E3_15300180012").setMaster("spark://10.190.2.112:7077")
sc = SparkContext(conf=conf)


def get_age_range(record):
    "for E3, to get age range from birth date"
    record = record.split('\t')
    birth_date_str = record[8]
    birth = birth_date_str.split('/')
    try:
        birth[2] = int(birth[2])
    except ValueError:
        birth[2] = 5000
    age = 2018 - birth[2]
    if age < 0:
        return 'censored', 1
    elif age <= 18:
        return '0-18', 1
    elif age <= 28:
        return '19-28', 1
    elif age <= 38:
        return '29-38', 1
    elif 49 <= age <= 55:
        return '49-55', 1
    elif 60 <= age:
        return '60-', 1
    else:
        return 'others', 1


# get remote data
data = sc.textFile("hdfs://10.190.2.112/data/data_dump.txt")\
    .map(get_age_range)\
    .reduceByKey(lambda a, b: a + b)\
    .collect()


print data

sc.stop()
