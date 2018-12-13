# dataframe

from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline

# spark = SparkSession \
#     .builder \
#     .appName("H2_15300180012") \
#     .master("spark://10.190.2.112:7077") \
#     .config("spark.executor.memory", "6g") \
#     .config("spark.executor.cores", 4) \
#     .config("spark.driver.memory", "2g") \
#     .getOrCreate()

# train = spark.read.load("hdfs://10.190.2.112/data/train_set.txt",
#   format="csv", sep="\t", inferSchema="true", header="false")
# val = spark.read.load("hdfs://10.190.2.112/data/val_set.txt",
#   format="csv", sep="\t", inferSchema="true", header="false")
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .master("local[8]") \
    .config("spark.driver.memory", "12g") \
    .getOrCreate()  # Gets an existing SparkSession, otherwise creates a new one

total = spark.read.load("../../../../val/part-00000",
    format="csv", sep="\t", inferSchema="true", header="false")
# val = spark.read.load("../../../../val/part-00002",
#   format="csv", sep="\t", inferSchema="true", header="false")
train, val = total.randomSplit([0.8, 0.2])
# total = train.union(val)

with open('H2_15300180012_output.txt', 'a') as f:
    f.write('\n \n')
    f.write('jq_H2_15300180012_output_naive_bayes\n')

# create label
indexer2 = StringIndexer(inputCol="_c6", outputCol="label")
# create features
indexer1 = StringIndexer(inputCol="_c2", outputCol="c22")
# One-hot encoder
encoder = OneHotEncoder(inputCol="c22", outputCol="features")

# create the trainer and set its parameters
with open('H2_15300180012_output.txt', 'a') as f:
    f.write('Smoothing parameter: {} \n'.format(1.0))

nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="label", featuresCol="features")

# create Pipeline
pipeline = Pipeline(stages=[indexer1, indexer2, encoder, nb])

# train the model
model = pipeline.fit(train)
# only for feature transform
# total = train.union(val)

# # create features
# indexer = StringIndexer(inputCol="_c2", outputCol="c22")
# indexer = indexer.fit(total)
# train = indexer.transform(train)
# val = indexer.transform(val)
# # create label
# indexer = StringIndexer(inputCol="_c6", outputCol="label")
# indexer = indexer.fit(total)
# train = indexer.transform(train)
# val = indexer.transform(val)

# # One-hot encoder
# encoder = OneHotEncoder(inputCol="c22", outputCol="c2")
# train = encoder.transform(train)
# val = encoder.transform(val)

# # create the trainer and set its parameters
# with open('H2_15300180012_output.txt', 'a') as f:
#     f.write('\n \n')
#     f.write('jq_H2_15300180012_output_naive_bayes\n')

# para = 1.0
# with open('H2_15300180012_output.txt', 'a') as f:
#     f.write('Smoothing parameter: {} \n'.format(para))

# nb = NaiveBayes(smoothing=para, modelType="multinomial", labelCol="label", featuresCol="c2")

# # train the model
# model = nb.fit(train)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")

predictions = model.transform(train)
accuracy = evaluator.evaluate(predictions)
with open('H2_15300180012_output.txt', 'a') as f:
    f.write('training accuracy: {} \n'.format(accuracy))

predictions = model.transform(val)
accuracy = evaluator.evaluate(predictions)
with open('H2_15300180012_output.txt', 'a') as f:
    f.write('validation accuracy: {} \n'.format(accuracy))

spark.stop()

