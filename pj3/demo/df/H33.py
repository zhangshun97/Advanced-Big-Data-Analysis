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

spark = SparkSession \
    .builder \
    .appName("H3_15300180012") \
    .master("spark://10.190.2.112:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", 4) \
    .config("spark.driver.memory", "12g") \
    .getOrCreate()

train = spark.read.load("hdfs://10.190.2.112/data/train_set.txt",
  format="csv", sep="\t", inferSchema="true", header="false")
val = spark.read.load("hdfs://10.190.2.112/data/val_set.txt",
  format="csv", sep="\t", inferSchema="true", header="false")
train = train.rdd.map(lambda x: [x[10], x[8], x[6], x[1]%1000]).toDF(["_c10", "_c8", "_c6", "_c1"])
val = val.rdd.map(lambda x: [x[10], x[8], x[6], x[1]%1000]).toDF(["_c10", "_c8", "_c6", "_c1"])
total = train.union(val)

# spark = SparkSession \
#     .builder \
#     .appName("Python Spark SQL basic example") \
#     .master("local[*]") \
#     .config("spark.driver.memory", "12g") \
#     .getOrCreate()  # Gets an existing SparkSession, otherwise creates a new one

# train = spark.read.load("../../../../val/part-00001",
#     format="csv", sep="\t", inferSchema="true", header="false")
# val = spark.read.load("../../../../val/part-00002",
#   format="csv", sep="\t", inferSchema="true", header="false")
# total = train.union(val)
# total = total.rdd.map(lambda x: [x[10], x[8], x[6], x[1]%1000]).toDF(["_c10", "_c8", "_c6", "_c1"])
# train, val = total.randomSplit([0.7, 0.3])

# create features
indexer = StringIndexer(inputCol="_c10", outputCol="c21")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)
indexer = StringIndexer(inputCol="_c8", outputCol="c23")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)
indexer = StringIndexer(inputCol="_c6", outputCol="c24")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)
# create label
indexer = StringIndexer(inputCol="_c1", outputCol="label")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)

# assemble
assembler = VectorAssembler(
    inputCols=["c21", "c23", "c24"],
    outputCol="c22")

train = assembler.transform(train)
val = assembler.transform(val)

# One-hot encoder
# encoder = OneHotEncoder(inputCol="c22", outputCol="c2")
# train = encoder.transform(train)
# val = encoder.transform(val)


# create the trainer and set its parameters
with open('H3_15300180012_output_df.txt', 'a') as f:
    f.write('\n \n')
    f.write('jq_H33_15300180012_output_naive_bayes\n')

para = 1.0
with open('H3_15300180012_output_df.txt', 'a') as f:
    f.write('Smoothing parameter: {} \n'.format(para))
nb = NaiveBayes(smoothing=para, modelType="multinomial", labelCol="label", featuresCol="c22")

# train the model
model = nb.fit(train)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")

predictions = model.transform(train)
accuracy = evaluator.evaluate(predictions)
with open('H3_15300180012_output_df.txt', 'a') as f:
    f.write('training accuracy: {} \n'.format(accuracy))
# print "Train set accuracy = " + str(accuracy)

predictions = model.transform(val)
accuracy = evaluator.evaluate(predictions)
with open('H3_15300180012_output_df.txt', 'a') as f:
    f.write('validation accuracy: {} \n'.format(accuracy))
# print "Val set accuracy = " + str(accuracy)

spark.stop()

