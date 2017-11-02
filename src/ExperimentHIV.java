import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.SubsetAccuracy;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;

import pum.kebi.mlc.cc.ClassifierChain;
import pum.kebi.mlc.cc.EnsembleOfClassifierChains;
import pum.kebi.mlc.common.InstanceBasedFMeasure;
import pum.kebi.mlc.common.MacroAccuracy;
import pum.kebi.mlc.ns.EnsembleOfNestedStackers;
import pum.kebi.mlc.ns.NestedStacking;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;


/**
 * @author Robin Senge [senge@mathematik.uni-marburg.de]
 *
 */
public class ExperimentHIV {

	private final static int TEN_CV = 1;
	private final static int BOOTSTRAP = 2;
        private final static int PERFORMANCE = 3;

	private static MultiLabelLearnerBase mlLearner = null;
	private static MultiLabelInstances dataset = null;
	private static AbstractClassifier baseClassifier = null;
	private static int numRuns = 0; 
	private static int scheme = PERFORMANCE;
	private static int[] labelIndexes = null;
	
	private static String mlMethod = null;
	private static String baseClassifierClass = null;
	private static String[] baseClassifierOptions = null;
	private static String path_to_dataset = null;
	
	private static final String OPTION_MULTI_LABEL_METHOD 		= "m";
	private static final String OPTION_BASE_CLASSIFIER_CLASS 	= "b";
	private static final String OPTION_BASE_CLASSIFIER_OPTIONS 	= "o";
	private static final String OPTION_DATASET		 			= "d";

	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {

		// ############# O P T I O N S #################

		Options options = new Options();
		options.addOption(OPTION_MULTI_LABEL_METHOD, true, "multi-label method: BR|CC|CCprob|NS|ECC|ENS|CLR");
		options.addOption(OPTION_BASE_CLASSIFIER_CLASS, true, "base classifier class");
		options.addOption(OPTION_BASE_CLASSIFIER_OPTIONS, true, "base classifier options");
		options.addOption(OPTION_DATASET, true, "path to dataset (without suffix)");
		
		CommandLineParser parser = new PosixParser();
		CommandLine cmd = parser.parse( options, args );
				
		File file = new File("results.txt");
		FileOutputStream fos = new FileOutputStream(file);
		PrintStream ps = new PrintStream(fos);
		/* command line use
		 * 
		 * 
		mlMethod = cmd.getOptionValue(OPTION_MULTI_LABEL_METHOD);
		baseClassifierClass = cmd.getOptionValue(OPTION_BASE_CLASSIFIER_CLASS);
		baseClassifierOptions = Utils.splitOptions(cmd.getOptionValue(OPTION_BASE_CLASSIFIER_OPTIONS));
		path_to_dataset = cmd.getOptionValue(OPTION_DATASET);*/
		
		
		mlMethod = "ECC";
		baseClassifierClass = "weka.classifiers.trees.RandomForest";
		//baseClassifierClass = "weka.classifiers.functions.Logistic";
	

		//set parameter only for RF
		baseClassifierOptions = Utils.splitOptions("-I 8 -depth 10");
		
		
		// ############# E V A L U A T I O N #################

		baseClassifier = (AbstractClassifier) AbstractClassifier.forName(baseClassifierClass, baseClassifierOptions);
		
		System.out.println("Multi-label method: " + mlMethod);
		System.out.println("Base classifier: " + baseClassifierClass + " " + Utils.joinOptions(baseClassifier.getOptions()));
		
		if(mlMethod.equalsIgnoreCase("BR")) {
			mlLearner = new BinaryRelevance(AbstractClassifier.makeCopy(baseClassifier));
		} 
		else if(mlMethod.equalsIgnoreCase("CC")) {
			mlLearner = new ClassifierChain(AbstractClassifier.makeCopy(baseClassifier));
		}
		else if(mlMethod.equalsIgnoreCase("CCprob")) {
			ClassifierChain cc = new ClassifierChain(AbstractClassifier.makeCopy(baseClassifier));
			cc.setPropagateProbabilities(true);
			mlLearner = cc;
		}
		else if(mlMethod.equalsIgnoreCase("NS")) {
			mlLearner = new NestedStacking(AbstractClassifier.makeCopy(baseClassifier));
		}
		else if(mlMethod.equalsIgnoreCase("NSprob")) {
			NestedStacking ns = new NestedStacking(AbstractClassifier.makeCopy(baseClassifier));
			ns.setPropagateProbabilities(true);
			mlLearner = ns;
		}
		else if(mlMethod.equalsIgnoreCase("ECC")) {
			ClassifierChain cc = new ClassifierChain(AbstractClassifier.makeCopy(baseClassifier));
			mlLearner = new EnsembleOfClassifierChains(cc);
		}
		else if(mlMethod.equalsIgnoreCase("ENS")) {
			NestedStacking ns = new NestedStacking(AbstractClassifier.makeCopy(baseClassifier));
			mlLearner = new EnsembleOfNestedStackers(ns);
		}
		else if(mlMethod.equalsIgnoreCase("CLR")) {
			mlLearner = new CalibratedLabelRanking(AbstractClassifier.makeCopy(baseClassifier));
		}
		else throw new RuntimeException("Multi-label method unknown: " + mlMethod);
		
                ArrayList<String> dataSets = new ArrayList();
//		dataSets.add("data/bibtex");
//                dataSets.add("data/bookmarks");
//                dataSets.add("data/CAL500");
//                dataSets.add("data/Corel5k");
//                dataSets.add("data/delicious");
//                dataSets.add("data/emotions");
//                dataSets.add("data/enron");
//                dataSets.add("data/flags");
//                dataSets.add("data/genbase");
//                dataSets.add("data/mediamill");
//                dataSets.add("data/medical");
                dataSets.add("data/NNRTI");
//                dataSets.add("data/scene");
//                dataSets.add("data/tmc2007");
                dataSets.add("data/yeast");
                
		for(String setname : dataSets) {
                    ps.append("Data: "+setname+"\r\n");
                    dataset = new MultiLabelInstances(setname + ".arff", setname + ".xml");
                    performanceMeasurement(dataset, 0, mlLearner, ps);
		}
	}

        private static void performanceMeasurement(MultiLabelInstances dataset, int r, MultiLabelLearner learner, PrintStream ps) throws Exception {
            int[] labelIndexes = dataset.getLabelIndices();
            Instances tmp = new Instances(dataset.getDataSet());
            tmp.randomize(new Random(r+1));
		
            MultiLabelInstances sample = new MultiLabelInstances(tmp , dataset.getLabelsMetaData());
            int trainInstCnt = (int)(sample.getNumInstances() * 0.67);
            int testInstCnt = sample.getNumInstances() - trainInstCnt;
           
            MultiLabelInstances mlTrain = new MultiLabelInstances(new Instances(sample.getDataSet(), 0, trainInstCnt), sample.getLabelsMetaData());
            MultiLabelInstances mlTest = new MultiLabelInstances(new Instances(sample.getDataSet(), trainInstCnt, testInstCnt), sample.getLabelsMetaData());

            MultiLabelLearner clone = learner.makeCopy();

            clone.build(mlTrain);
            
            MultiLabelOutput[] output = new MultiLabelOutput[mlTest.getNumInstances()];

            System.out.println("Classification:");
            long start = System.nanoTime();
            for(int j = 0; j < mlTest.getNumInstances(); j++) {
                Instance instance = mlTest.getDataSet().get(j);
                output[j] = clone.makePrediction(instance);
            }
            double time = (double)(System.nanoTime() - start)*1e-06;
            System.out.println(time + " ms");
            ps.append("Took: "+time+" ms\r\n");
            
            int hits = 0;
            for(int i = 0; i < mlTest.getNumInstances(); i++) {
                double[] confidences = output[i].getConfidences();
                Instance instance = mlTest.getDataSet().get(i);
                for(int l = 0; l < mlTest.getNumLabels(); l++){
                    boolean pred = confidences[l] > 0.5;
                    boolean real = instance.value(mlTest.getLabelIndices()[l]) > 0.5;
                    if(pred == real)
                        hits++;
                }
            }
            System.out.println("Prediction: " + (double)hits / (double)(mlTest.getNumInstances()*mlTest.getNumLabels()) + "%");
            ps.append("Prediction: " + (double)hits / (double)(mlTest.getNumInstances()*mlTest.getNumLabels()) + "%\r\n\r\n");
        }
}
