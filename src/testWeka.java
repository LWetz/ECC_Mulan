
import java.io.FileReader;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class testWeka {

    public static void main(String[] args) throws Exception {
        String arffFilename = Utils.getOption("arff", args);
        String xmlFilename = Utils.getOption("xml", args);

    	//String arffFilename = "emotions.arff";
    	//String xmlFilename = "emotions.xml";
    	
    	
        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

        System.out.println(dataset.getClass());
        
        RAkEL model = new RAkEL(new LabelPowerset(new J48()));
        System.out.println(model.getClass());
        model.build(dataset);


        //String unlabeledFilename = Utils.getOption("unlabeled", args);
        String unlabeledFilename = arffFilename;
        FileReader reader = new FileReader(unlabeledFilename);
        Instances unlabeledData = new Instances(reader);

        int numInstances = unlabeledData.numInstances();

        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = unlabeledData.instance(instanceIndex);
            MultiLabelOutput output = model.makePrediction(instance);
            // do necessary operations with provided prediction output, here just print it out
            System.out.println(output);
        }
    }
}

