# RDD-based

from pyspark.sql import SparkSession
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
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
test = spark.read.load("hdfs://10.190.2.112/data/val_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")

def gender(x):
	if x[6] == 'E':
		return 1
	else:
		return 0

first_names = train.rdd.map(lambda x: x[2]).distinct().collect()
_ind = 1
name_dict = dict()
for name in first_names:
	name_dict[name] = _ind
	_ind += 1
print _ind
# create features and labels

train = train.rdd.map(lambda x: LabeledPoint(gender(x), [name_dict.get(x[2], 0)]))
test = test.rdd.map(lambda x: LabeledPoint(gender(x), [name_dict.get(x[2], 0)]))

# Train a naive Bayes model.
model = NaiveBayes.train(train, 1.0)

# Make prediction and test accuracy.
predictionAndLabel = train.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / train.count()
print 'model accuracy {}'.format(accuracy)

# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print 'model accuracy {}'.format(accuracy)

# Save and load model
# output_dir = 'target/tmp/myNaiveBayesModel'
# shutil.rmtree(output_dir, ignore_errors=True)
# model.save(sc, output_dir)
# sameModel = NaiveBayesModel.load(sc, output_dir)
# predictionAndLabel = test.map(lambda p: (sameModel.predict(p.features), p.label))
# accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
# print 'sameModel accuracy {}'.format(accuracy)

spark.stop()
