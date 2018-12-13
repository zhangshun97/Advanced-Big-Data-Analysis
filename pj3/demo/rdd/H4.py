from pyspark.sql import SparkSession
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import HashingTF

# initialization SparkSession
spark = SparkSession \
    .builder \
    .appName("H2_15300180012") \
    .master("spark://10.190.2.112:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.executor.cores", 4) \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# spark = SparkSession \
#     .builder \
#     .appName("Python Spark SQL basic example") \
#     .master("local[4]") \
#     .getOrCreate()

# Load training data
train = spark.read.load("hdfs://10.190.2.112/data/train_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")
val = spark.read.load("hdfs://10.190.2.112/data/val_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")
test = spark.read.load("hdfs://10.190.2.112/data/test_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")
# train = spark.read.load("./val/part-00000",
# 	format="csv", sep="\t", inferSchema="true", header="false")
# val = spark.read.load("./val/part-00001",
# 	format="csv", sep="\t", inferSchema="true", header="false")
# test = spark.read.load("./val/part-00002",
# 	format="csv", sep="\t", inferSchema="true", header="false")
total = train.union(val).union(test)
# Extract features
last_names = total.rdd.map(lambda x: x[3]).distinct().collect()
_ind = 1
name_dict = dict()
for name in last_names:
	name_dict[name] = _ind
	_ind += 1

# districts = train.rdd.flatMap(lambda x: x[12]).distinct().collect()
# _ind = 1
# district_dict = dict()
# for district in districts:
# 	district_dict[district] = _ind
# 	_ind += 1
# create features and labels
HDF = HashingTF(50)
train = train.rdd.map(lambda x: LabeledPoint(name_dict.get(x[3], 0), HDF.transform(x[12])))
test = test.rdd.map(lambda x: LabeledPoint(name_dict.get(x[3], 0), HDF.transform(x[12])))
val = val.rdd.map(lambda x: LabeledPoint(name_dict.get(x[3], 0), HDF.transform(x[12])))


with open('H4_15300180012_output.txt', 'w') as f:
	f.write('H4_15300180012_output_naive_bayes\n')

para = 1.0
with open('H4_15300180012_output.txt', 'a') as f:
	f.write('Smoothing parameter: {} \n'.format(para))

# Train a naive Bayes model.
model = NaiveBayes.train(train, para)

# train accuracy.
predictionAndLabel = train.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / train.count()
with open('H4_15300180012_output.txt', 'a') as f:
	f.write('training accuracy: {} \n'.format(accuracy))
# print 'model accuracy {}'.format(accuracy)

# validation accuracy.
predictionAndLabel = val.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / val.count()
with open('H4_15300180012_output.txt', 'a') as f:
	f.write('validation accuracy: {} \n'.format(accuracy))
# print 'model accuracy {}'.format(accuracy)

# test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
print predictionAndLabel.collect
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
with open('H4_15300180012_output.txt', 'a') as f:
	f.write('test accuracy: {} \n'.format(accuracy))
# print 'model accuracy {}'.format(accuracy)


para = 0.1
with open('H4_15300180012_output.txt', 'a') as f:
	f.write('Regularization(L2) parameter: {} \n'.format(para))

# Train a Logistic Regression model.
model = LogisticRegressionWithLBFGS.train(train, iterations=100, regParam=para, regType='l2')

# train accuracy.
predictionAndLabel = train.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / train.count()
with open('H4_15300180012_output.txt', 'a') as f:
	f.write('training accuracy: {} \n'.format(accuracy))
# print 'model accuracy {}'.format(accuracy)

# validation accuracy.
predictionAndLabel = val.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / val.count()
with open('H4_15300180012_output.txt', 'a') as f:
	f.write('validation accuracy: {} \n'.format(accuracy))
# print 'model accuracy {}'.format(accuracy)

# test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
print predictionAndLabel.collect
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
with open('H4_15300180012_output.txt', 'a') as f:
	f.write('test accuracy: {} \n'.format(accuracy))
# print 'model accuracy {}'.format(accuracy)

spark.stop()


