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
    .appName("H4_15300180012") \
    .master("spark://10.190.2.112:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.executor.cores", 4) \
    .config("spark.driver.memory", "12g") \
    .getOrCreate()

train = spark.read.load("hdfs://10.190.2.112/data/train_set.txt",
  format="csv", sep="\t", inferSchema="true", header="false")
val = spark.read.load("hdfs://10.190.2.112/data/val_set.txt",
  format="csv", sep="\t", inferSchema="true", header="false")

# spark = SparkSession \
#     .builder \
#     .appName("Python Spark SQL basic example") \
#     .master("local[*]") \
#     .config("spark.driver.memory", "12g") \
#     .getOrCreate()  # Gets an existing SparkSession, otherwise creates a new one

# train = spark.read.load("../../../../val/part-00000",
#     format="csv", sep="\t", inferSchema="true", header="false")
# val = spark.read.load("../../../../val/part-00002",
#   format="csv", sep="\t", inferSchema="true", header="false")

total = train.union(val)

total.createOrReplaceTempView("temp")
total = spark.sql("SELECT * FROM temp WHERE _c3 in ( \
                    SELECT _c3 FROM temp \
                    GROUP BY _c3 \
                    ORDER BY COUNT(*) DESC \
                    LIMIT 20)")
train, val = total.randomSplit([0.8, 0.2])

# create features
indexer = StringIndexer(inputCol="_c12", outputCol="c22")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)
# create label
indexer = StringIndexer(inputCol="_c3", outputCol="label")
indexer = indexer.fit(total)
train = indexer.transform(train)
val = indexer.transform(val)

# One-hot encoder
encoder = OneHotEncoder(inputCol="c22", outputCol="c2")
train = encoder.transform(train)
val = encoder.transform(val)

total.unpersist()
# assemble
# assembler = VectorAssembler(
#     inputCols=["c2"],
#     outputCol="features")

# train = assembler.transform(train)
# test = assembler.transform(test)

# create the trainer and set its parameters
with open('H4_15300180012_output_df.txt', 'a') as f:
    f.write('\n \n')
    f.write('jq_H4_15300180012_output_naive_bayes\n')

para = 1.0
with open('H4_15300180012_output_df.txt', 'a') as f:
    f.write('Smoothing parameter: {} \n'.format(para))
nb = NaiveBayes(smoothing=para, modelType="multinomial", labelCol="label", featuresCol="c2")

# train the model
model = nb.fit(train)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")

predictions = model.transform(train)
accuracy = evaluator.evaluate(predictions)
with open('H4_15300180012_output_df.txt', 'a') as f:
    f.write('training accuracy: {} \n'.format(accuracy))
# print "Train set accuracy = " + str(accuracy)

predictions = model.transform(val)
accuracy = evaluator.evaluate(predictions)
with open('H4_15300180012_output_df.txt', 'a') as f:
    f.write('validation accuracy: {} \n'.format(accuracy))
# print "Val set accuracy = " + str(accuracy)

spark.stop()

