# H2: predict gender based on one's first name

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
    .appName("H2_15300180012") \
    .master("spark://10.190.2.112:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.executor.cores", 4) \
    .config("spark.driver.memory", "12g") \
    .getOrCreate()

train = spark.read.load("hdfs://10.190.2.112/data/train_set.txt",
  format="csv", sep="\t", inferSchema="true", header="false")
val = spark.read.load("hdfs://10.190.2.112/data/val_set.txt",
  format="csv", sep="\t", inferSchema="true", header="false")
test = spark.read.load("hdfs://10.190.2.112/data/test_set.txt",
  format="csv", sep="\t", inferSchema="true", header="false")

# only for feature transform
total = train.union(val).union(test)

# create features
indexer = StringIndexer(inputCol="_c2", outputCol="c22")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)
test = indexer.transform(test)
# create label
indexer = StringIndexer(inputCol="_c6", outputCol="label")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)
test = indexer.transform(test)
# One-hot encoder
encoder = OneHotEncoder(inputCol="c22", outputCol="c2")
train = encoder.transform(train)
val = encoder.transform(val)
test = encoder.transform(test)

# create the trainer and set its parameters
with open('H2_15300180012_output.txt', 'a') as f:
    f.write('\n \n')
    f.write('jq_H2_15300180012_output_naive_bayes\n')

para = 1.0
with open('H2_15300180012_output.txt', 'a') as f:
    f.write('Smoothing parameter: {} \n'.format(para))

nb = NaiveBayes(smoothing=para, modelType="multinomial", labelCol="label", featuresCol="c2")

# train the model
model = nb.fit(train)

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

predictions = model.transform(test)
accuracy = evaluator.evaluate(predictions)
with open('H2_15300180012_output.txt', 'a') as f:
    f.write('test accuracy: {} \n'.format(accuracy))

spark.stop()

