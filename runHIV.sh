
java -Xmx2048M -classpath weka.jar:mulan.jar:mlc.jar:cli.jar ExperimentHIV \
	-m BR \
	-b "weka.classifiers.trees.RandomForest" \
	-o "-I 10" \
	-d "/Users/dicmen/Dropbox/Work/Uni/Experimente/HIV/Daten3"