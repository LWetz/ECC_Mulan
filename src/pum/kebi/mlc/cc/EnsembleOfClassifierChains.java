/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    EnsembleOfClassifierChains.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package pum.kebi.mlc.cc;

import java.util.Arrays;
import java.util.Random;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 <!-- globalinfo-start -->
 * Class implementing the Ensemble of Classifier Chains(ECC) algorithm. For more information, see<br/>
 * <br/>
 * Read, Jesse, Pfahringer, Bernhard, Holmes, Geoff, Frank, Eibe: Classifier Chains for Multi-label Classification. In: , 335--359, 2011.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Read2011,
 *    author = {Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff and Frank, Eibe},
 *    journal = {Machine Learning},
 *    number = {3},
 *    pages = {335--359},
 *    title = {Classifier Chains for Multi-label Classification},
 *    volume = {85},
 *    year = {2011}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * @author Konstantinos Sechidis (sechidis@csd.auth.gr)
 * @author Grigorios Tsoumakas (greg@csd.auth.gr)
 * @author Robin Senge (senge@mathematik.uni-marburg.de)
 * @version 2012.02.27
 */
public class EnsembleOfClassifierChains extends MultiLabelLearnerBase {

	
	private static final long serialVersionUID = -2157472669259464267L;


	/** The base classifier chain models to use for sampling. */
    protected ClassifierChain baseClassifierChain;
	
	/** An array of ClassifierChain models. */
    protected ClassifierChain[] ensemble;
    
    /** The number of classifier chain models. */
    protected int ensembleSize = 8;
    
    /** Random number generator. */
    protected Random rand;
    
    /**
     * Whether the output is computed based on the average votes or on the
     * average confidences
     */
    protected boolean softVoting;
    
    /**
     * Whether to use sampling with replacement to create the data of the models
     * of the ensemble
     */
    protected boolean useSamplingWithReplacement = true;
    
    /**
     * The size of each bag sample, as a percentage of the training size. Used
     * when useSamplingWithReplacement is true.
     */
    protected int bagSizePercent = 100;

    /**
     * The size of each sample, as a percentage of the training size Used when
     * useSamplingWithReplacement is false
     */
    protected double samplingPercentage = 67;

    
    /**
     * Creates a new ensemble of classifier chains.
     */
    public EnsembleOfClassifierChains(ClassifierChain baseClassifierChain) {
        this.baseClassifierChain = baseClassifierChain;
        ensemble = new ClassifierChain[ensembleSize];
        rand = new Random();
    }

    /**
     * Returns a string describing classifier.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {
        return "Class implementing the Ensemble of Classifier Chains"
                + "(ECC) algorithm. For more information, see\n\n"
                + getTechnicalInformation().toString();
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff and Frank, Eibe");
        result.setValue(Field.TITLE, "Classifier Chains for Multi-label Classification");
        result.setValue(Field.VOLUME, "85");
        result.setValue(Field.NUMBER, "3");
        result.setValue(Field.YEAR, "2011");
        result.setValue(Field.PAGES, "335--359");
        result.setValue(Field.JOURNAL, "Machine Learning");
        return result;
    }

    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {

        Instances dataSet = new Instances(trainingSet.getDataSet());

        for (int i = 0; i < ensembleSize; i++) {
            debug("ECC Building Model:" + (i + 1) + "/" + ensembleSize);
            Instances sampledDataSet;
            dataSet.randomize(rand);
            if (useSamplingWithReplacement) {
                int bagSize = dataSet.numInstances() * bagSizePercent / 100;
                // create the in-bag dataset
                sampledDataSet = dataSet.resampleWithWeights(new Random(1));
                if (bagSize < dataSet.numInstances()) {
                    sampledDataSet = new Instances(sampledDataSet, 0, bagSize);
                }
            } else {
                RemovePercentage rmvp = new RemovePercentage();
                rmvp.setInvertSelection(true);
                rmvp.setPercentage(samplingPercentage);
                rmvp.setInputFormat(dataSet);
                sampledDataSet = Filter.useFilter(dataSet, rmvp);
            }
            MultiLabelInstances train = new MultiLabelInstances(sampledDataSet, trainingSet.getLabelsMetaData());

            int[] chain = new int[numLabels];
            for (int j = 0; j < numLabels; j++) {
                chain[j] = j;
            }
            for (int j = 0; j < chain.length; j++) {
                int randomPosition = rand.nextInt(chain.length);
                int temp = chain[j];
                chain[j] = chain[randomPosition];
                chain[randomPosition] = temp;
            }
            debug(Arrays.toString(chain));

            ensemble[i] = (ClassifierChain)baseClassifierChain.makeCopy();
            ensemble[i].setChain(chain);
            ensemble[i].build(train);
        }

    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {

        int[] sumVotes = new int[numLabels];
        double[] sumConf = new double[numLabels];

        Arrays.fill(sumVotes, 0);
        Arrays.fill(sumConf, 0);

        for (int i = 0; i < ensembleSize; i++) {
            MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);
            boolean[] bip = ensembleMLO.getBipartition();
            double[] conf = ensembleMLO.getConfidences();

            for (int j = 0; j < numLabels; j++) {
                sumVotes[j] += bip[j] == true ? 1 : 0;
                sumConf[j] += conf[j];
            }
        }

        double[] confidence = new double[numLabels];
        for (int j = 0; j < numLabels; j++) {
            if (softVoting) {
                confidence[j] = sumConf[j] / ensembleSize;
            } else {
                confidence[j] = sumVotes[j] / (double) ensembleSize;
            }
        }

        MultiLabelOutput mlo = new MultiLabelOutput(confidence, 0.5);
        return mlo;
    }
    
    
    public ClassifierChain getBaseClassifierChainModel() {
		return baseClassifierChain;
	}
    
    public int getBagSizePercent() {
        return bagSizePercent;
    }

    public void setBagSizePercent(int bagSizePercent) {
        this.bagSizePercent = bagSizePercent;
    }

    public double getSamplingPercentage() {
        return samplingPercentage;
    }

    public void setSamplingPercentage(double samplingPercentage) {
        this.samplingPercentage = samplingPercentage;
    }
    
    public void setEnsembleSize(int ensembleSize) {
		this.ensembleSize = ensembleSize;
	}
    
    public int getEnsembleSize() {
		return ensembleSize;
	}
    
    
}