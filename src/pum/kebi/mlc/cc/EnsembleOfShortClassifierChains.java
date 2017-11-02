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
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Ensemble of chort classifier chains.
 * 
 * @author Robin Senge [senge@mathematik.uni-marburg.de]
 */
public class EnsembleOfShortClassifierChains extends MultiLabelLearnerBase {

    private static final long serialVersionUID = -1015244136309433182L;

	/** The number of classifier chain models. */
    protected int ensembleSize;
    
    /** Base classifier chain model. */
    protected ClassifierChain baseClassifierChain;
    
    /** An array of ClassifierChain models. */
    protected ClassifierChain[] ensemble;
    
    /** The chains, the single models are trained on. */
    protected int[][] chains = null; 
    
    /** The chains, but sorted. */
    protected int[][] sortedChains = null;
    
    /** Random number generator. */
    protected Random rand;
    
    /** The maximum length of a label chain. */
    protected int maxChainLength;
    
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
     * when useSamplingWithReplacement is true
     */
    protected int BagSizePercent = 100;

    /**
     * The size of each sample, as a percentage of the training size Used when
     * useSamplingWithReplacement is false
     */
    protected double samplingPercentage = 67;

    
    /**
     * Creates a new ensemble of short classifier chains.
     */
    public EnsembleOfShortClassifierChains(ClassifierChain baseClassifierChain) {
        this.baseClassifierChain = baseClassifierChain;
    }

    @Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	/**
     * Returns a string describing classifier.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    @Override
    public String globalInfo() {
        return "Class implementing the Ensemble of Short Classifier Chains"
                + "(ESCC) algorithm. For more information, see";
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {

    	this.maxChainLength = Math.min(maxChainLength, trainingSet.getNumLabels());
    	if(this.maxChainLength == trainingSet.getNumLabels()) 
    		debug("Chain length equals the number of labels of the dataset." +
    				" This comes down to standard ECC, then!");
    	if(this.maxChainLength * this.ensembleSize < trainingSet.getNumLabels()) 
    		throw new IllegalArgumentException("Not all labels are covered by at least one chain! " +
    				"Increase the number of models or the chain length!");
    	
    	Instances dataSet = new Instances(trainingSet.getDataSet());
        
    	// create random chains born from the mother chain ;-)
    	Random rand = new Random(1);
    	List<Integer> mother = new LinkedList<Integer>(); 
        for (int j = 0; j < numLabels; j++) {
            mother.add(j);
        }
        this.chains = new int[ensembleSize][maxChainLength];
        this.sortedChains = new int[ensembleSize][];
        for (int i = 0; i < ensembleSize; i++) {
			List<Integer> tmp = new LinkedList<Integer>(mother);
			for(int j = 0; j < maxChainLength; j++) {
				int t = rand.nextInt(tmp.size());
				this.chains[i][j] = tmp.get(t);
				tmp.remove(t);
			}
			sortedChains[i] = Arrays.copyOf(chains[i], maxChainLength);
			Arrays.sort(sortedChains[i]);
		}
        
        
        // build models
        for (int i = 0; i < ensembleSize; i++) {
            debug("ESCC Building Model:" + (i + 1) + "/" + ensembleSize);
            Instances sampledDataSet;
            dataSet.randomize(rand);
            if (useSamplingWithReplacement) {
                int bagSize = dataSet.numInstances() * BagSizePercent / 100;
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

            ensemble[i] = (ClassifierChain)baseClassifierChain.makeCopy();
            ensemble[i].setChain(chains[i]);
            ensemble[i].build(train);
        }

    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {

        int[] sumVotes = new int[numLabels];
        int[] numModels = new int[numLabels];
        double[] sumConf = new double[numLabels];
        
        Arrays.fill(sumVotes, 0);
        Arrays.fill(sumConf, 0);

        for (int i = 0; i < ensembleSize; i++) {
            MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);
            boolean[] bip = ensembleMLO.getBipartition();
            double[] conf = ensembleMLO.getConfidences();

            for (int j = 0; j < numLabels; j++) {
            	if(!Double.isNaN(conf[j])) {
            		numModels[j]++;
            		sumVotes[j] += bip[j] ? 1 : 0;
            		sumConf[j] += conf[j];
            	}
            }
        }
        
        // check, if every label had at least one model
        for(int i = 0; i < numLabels; i++) {
        	if(numModels[i] == 0)
        		throw new RuntimeException(String.format("Label %s without model! Improve implementation!", i));
        }

        double[] confidence = new double[numLabels];
        for (int j = 0; j < numLabels; j++) {
            if (softVoting) {
                confidence[j] = sumConf[j] / numModels[j];
            } else {
                confidence[j] = sumVotes[j] / (double) numModels[j];
            }
        }

        MultiLabelOutput mlo = new MultiLabelOutput(confidence, 0.5);
        return mlo;
    }

	public int getBagSizePercent() {
        return BagSizePercent;
    }

    public void setBagSizePercent(int bagSizePercent) {
        BagSizePercent = bagSizePercent;
    }

    public double getSamplingPercentage() {
        return samplingPercentage;
    }

    public void setSamplingPercentage(double samplingPercentage) {
        this.samplingPercentage = samplingPercentage;
    }
    
    public void setMaxChainLength(int maxChainLength) {
		this.maxChainLength = maxChainLength;
	}
    
    public int getMaxChainLength() {
		return maxChainLength;
	}
    
    public void setEnsembleSize(int ensembleSize) {
		this.ensembleSize = ensembleSize;
	}
    
    public int getEnsembleSize() {
		return ensembleSize;
	}
	
}