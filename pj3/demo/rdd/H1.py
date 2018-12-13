from pyspark.sql import SparkSession
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import HashingTF
from pyspark.ml.feature import StringIndexer

# initialization SparkSession
spark = SparkSession \
    .builder \
    .appName("H1_15300180012") \
    .master("spark://10.190.2.112:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.executor.cores", 4) \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load training data
train = spark.read.load("hdfs://10.190.2.112/data/train_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")
val = spark.read.load("hdfs://10.190.2.112/data/val_set.txt",
	format="csv", sep="\t", inferSchema="true", header="false")

# spark = SparkSession \
#     .builder \
#     .appName("Python Spark SQL basic example") \
#     .master("local[8]") \
#     .getOrCreate()  # Gets an existing SparkSession, otherwise creates a new one

# train = spark.read.load("../../../../val/part-00000",
#     format="csv", sep="\t", inferSchema="true", header="false")
# val = spark.read.load("../../../../val/part-00002",
#   format="csv", sep="\t", inferSchema="true", header="false")

total = train.union(val)

# indexer = StringIndexer(inputCol="_c11", outputCol="label")
# indexer = indexer.fit(total)
# train = indexer.transform(train)
# val = indexer.transform(val)
# indexer = StringIndexer(inputCol="_c12", outputCol="features")
# indexer = indexer.fit(total)
# train = indexer.transform(train)
# val = indexer.transform(val)

# Extract features
cities = total.rdd.map(lambda x: x[11]).distinct().collect()
districts = total.rdd.map(lambda x: x[12]).distinct().collect()
_ind = 1
city_dict = dict()
for city in cities:
	city_dict[city] = _ind
	_ind += 1
_ind = 1
district_dict = dict()
for district in districts:
	district_dict[district] = _ind
	_ind += 1
# create features and labels

train = train.rdd.map(lambda x: LabeledPoint(city_dict.get(x[11], 0), [district_dict.get(x[12], 0)]))
val = val.rdd.map(lambda x: LabeledPoint(city_dict.get(x[11], 0), [district_dict.get(x[12], 0)]))

# HDF = HashingTF(50)
# train = train.rdd.map(lambda x: LabeledPoint(x[-2], [x[-1]]))
# val = val.rdd.map(lambda x: LabeledPoint(x[-2], [x[-1]]))

with open('H1_15300180012_output_nb.txt', 'w') as f:
	f.write('H1_15300180012_output_naive_bayes\n')

def do_training(para=1.0):
	with open('H1_15300180012_output_nb.txt', 'a') as f:
		f.write('Regularization(L2) parameter: {} \n'.format(para))

	# Train a naive Bayes model.
	model = NaiveBayes.train(train, para)

	# train accuracy.
	predictionAndLabel = train.map(lambda p: (model.predict(p.features), p.label))
	accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / train.count()
	with open('H1_15300180012_output_nb.txt', 'a') as f:
		f.write('training accuracy: {} \n'.format(accuracy))
	# print 'model accuracy {}'.format(accuracy)

	# validation accuracy.
	predictionAndLabel = val.map(lambda p: (model.predict(p.features), p.label))
	accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / val.count()
	with open('H1_15300180012_output_nb.txt', 'a') as f:
		f.write('validation accuracy: {} \n'.format(accuracy))
	# print 'model accuracy {}'.format(accuracy)

for para in [1.0]:
	do_training(para)

spark.stop()

