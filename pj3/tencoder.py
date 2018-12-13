# test for Spark
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .master("local[4]") \
    .getOrCreate()  # Gets an existing SparkSession, otherwise creates a new one

train = spark.read.load("./val/part-00000",
	format="csv", sep="\t", inferSchema="true", header="false")

indexer = StringIndexer(inputCol="_c2", outputCol="c22")
train = indexer.fit(train).transform(train)
encoder = OneHotEncoder(inputCol="c22", outputCol="c2")
train = encoder.transform(train)

splits = train.randomSplit([0.7,0.1,0.2])
train = splits[0]
val = splits[1]
test = splits[2]



# create features and labels
# indexer = StringIndexer(inputCol="_c2", outputCol="c22")
# indexer = indexer.fit(total)
# train = indexer.transform(train)
# test = indexer.transform(test)
# val = indexer.transform(val)
# indexer = StringIndexer(inputCol="_c6", outputCol="label")
# indexer = indexer.fit(total)
# train = indexer.transform(train)
# test = indexer.transform(test)
# val = indexer.transform(val)

# One-hot encoder
# encoder = OneHotEncoder(inputCol="c22", outputCol="c2")
# train = encoder.transform(train)
# test = encoder.transform(test)
# val = encoder.transform(val)
a1 = train.rdd.map(lambda x: x[2]).distinct().count()
a2 = val.rdd.map(lambda x: x[2]).distinct().count()
a3 = test.rdd.map(lambda x: x[2]).distinct().count()
train.show()
test.show()
print 'train', a1
print 'val', a2
print 'test', a3

spark.stop()

