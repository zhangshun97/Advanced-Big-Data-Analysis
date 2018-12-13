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
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# spark = SparkSession \
#     .builder \
#     .appName("H1_15300180012") \
#     .master("spark://10.190.2.112:7077") \
#     .config("spark.executor.memory", "6g") \
#     .config("spark.executor.cores", 4) \
#     .config("spark.driver.memory", "12g") \
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

# create features
indexer = StringIndexer(inputCol="_c9", outputCol="c21")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)
indexer = StringIndexer(inputCol="_c11", outputCol="c23")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)
# create label
indexer = StringIndexer(inputCol="_c7", outputCol="label")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)

total.unpersist()

# assemble
assembler = VectorAssembler(
    inputCols=["c21", "c23"],
    outputCol="c22")

train = assembler.transform(train)
val = assembler.transform(val)

# One-hot encoder
# encoder = OneHotEncoder(inputCol="c22", outputCol="c2")
# train = encoder.transform(train)
# val = encoder.transform(val)

# create the trainer and set its parameters
with open('H1_15300180012_output_df.txt', 'a') as f:
    f.write('\n \n')
    f.write('H1_15300180012_output_naive_bayes_birth_top_5\n')

para = 1.0
with open('H1_15300180012_output_df.txt', 'a') as f:
    f.write('Smoothing parameter: {} \n'.format(para))
nb = NaiveBayes(smoothing=para, modelType="multinomial", labelCol="label", featuresCol="c22")

# train the model
model = nb.fit(train)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")

# Top-5 accuracy
def if_in_top_n(label, probability_list):
    import numpy as np  # only required in this function
    probability_np = np.array(probability_list.values)
    n_ge_label = np.sum(probability_np > probability_np[int(label)])
    return n_ge_label < 5

def evaluator_top_n(predictions, n):
    pred = predictions.select(predictions.label, predictions.probability)
    if_top = udf(if_in_top_n, IntegerType())
    new_pred = pred.withColumn("top-{}".format(n), if_top("label", "probability"))
    accuracy = new_pred.filter(new_pred["top-{}".format(n)] == 1).count() / new_pred.count()
    return accuracy

# Top-1 accuracy
predictions = model.transform(train)
accuracy = evaluator.evaluate(predictions)
with open('H1_15300180012_output_df.txt', 'a') as f:
    f.write('training accuracy: {} \n'.format(accuracy))

accuracy2 = evaluator_top_n(predictions, 5)
with open('H1_15300180012_output_df.txt', 'a') as f:
    f.write('top-5 training accuracy: {} \n'.format(accuracy2))

predictions = model.transform(val)
accuracy = evaluator.evaluate(predictions)
with open('H1_15300180012_output_df.txt', 'a') as f:
    f.write('validation accuracy: {} \n'.format(accuracy))

accuracy2 = evaluator_top_n(predictions, 5)
with open('H1_15300180012_output_df.txt', 'a') as f:
    f.write('top-5 validation accuracy: {} \n'.format(accuracy2))

spark.stop()

