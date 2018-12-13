# RDD-based

from pyspark.sql import SparkSession
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import HashingTF


spark = SparkSession \
    .builder \
    .appName("H2_15300180012") \
    .master("spark://10.190.2.112:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.executor.cores", 4) \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

train = spark.read.load("hdfs://10.190.2.112/data/train_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")
val = spark.read.load("hdfs://10.190.2.112/data/val_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")
test = spark.read.load("hdfs://10.190.2.112/data/test_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")

# create features and labels
HDF = HashingTF(50)
train = train.rdd.map(lambda x: LabeledPoint(x[6] == 'E', HDF.transform([x[2], x[3]])))
test = test.rdd.map(lambda x: LabeledPoint(x[6] == 'E', HDF.transform([x[2], x[3]])))
val = val.rdd.map(lambda x: LabeledPoint(x[6] == 'E', HDF.transform([x[2], x[3]])))
# train = train.rdd.map(lambda x: LabeledPoint(gender(x), HDF.transform(x[2])))
# test = test.rdd.map(lambda x: LabeledPoint(gender(x), HDF.transform(x[2])))

with open('H2_15300180012_output_lr.txt', 'w') as f:
	f.write('H2_15300180012_output_logistic_regression\n')

def do_training(para=1.0):
	with open('H2_15300180012_output_lr.txt', 'a') as f:
		f.write('Regularization(L2) parameter: {} \n'.format(para))

	# Train a Logistic Regression model.
	model = LogisticRegressionWithLBFGS.train(train, iterations=100, regParam=para, regType='l2')

	# train accuracy.
	predictionAndLabel = train.map(lambda p: (model.predict(p.features), p.label))
	accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / train.count()
	with open('H2_15300180012_output_lr.txt', 'a') as f:
		f.write('training accuracy: {} \n'.format(accuracy))
	# print 'model accuracy {}'.format(accuracy)

	# validation accuracy.
	predictionAndLabel = val.map(lambda p: (model.predict(p.features), p.label))
	accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / val.count()
	with open('H2_15300180012_output_lr.txt', 'a') as f:
		f.write('validation accuracy: {} \n'.format(accuracy))
	# print 'model accuracy {}'.format(accuracy)

	# test accuracy.
	predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
	accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
	with open('H2_15300180012_output_lr.txt', 'a') as f:
		f.write('test accuracy: {} \n'.format(accuracy))
	# print 'model accuracy {}'.format(accuracy)


for para in [0.01, 0.1, 1, 10]:
	do_training(para)

spark.stop()
