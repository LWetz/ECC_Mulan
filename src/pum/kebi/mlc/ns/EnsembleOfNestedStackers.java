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
package pum.kebi.mlc.ns;

import java.util.Arrays;
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
 * @author Robin Senge [senge@mathematik.uni-marburg.de]
 */
public class EnsembleOfNestedStackers extends MultiLabelLearnerBase {

	private static final long serialVersionUID = -3103170092336923432L;

	/** The number of nested stacking models. */
    protected int ensembleSize;
    
    /** The base nested stacking classifier. */
    protected NestedStacking baseNestedStacking;
    
    /** An array of NestedStacking models. */
    protected NestedStacking[] ensemble;
    
    /** Random number generator. */
    protected Random rand;
    
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
     * If true, the confidence of each model is taken as soft vote.
     */
    protected boolean softVoting = false;
    
    /**
     * The size of each sample, as a percentage of the training size Used when
     * useSamplingWithReplacement is false
     */
    protected double samplingPercentage = 67;

    
    /** Creates a new ensemble of nested stackers. */
    public EnsembleOfNestedStackers(NestedStacking baseNestedStacking) {
        this.baseNestedStacking = baseNestedStacking;
    }

    /**
     * Returns a string describing classifier.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    @Override
    public String globalInfo() {
        return "Class implementing the Ensemble of Nested Stackers"
                + "(ENS) algorithm. For more information.\n"+
        		getTechnicalInformation().toString();
    }

    @Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {

        Instances dataSet = new Instances(trainingSet.getDataSet());

        for (int i = 0; i < ensembleSize; i++) {
            debug("ENS Building Model:" + (i + 1) + "/" + ensembleSize);
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

            ensemble[i] = (NestedStacking)baseNestedStacking.makeCopy();
            ensemble[i].setLabelOrder(chain);
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

	/** Returns the size of each bag sample, as a percentage of the training size. */
	public int getBagSizePercent() {
	    return BagSizePercent;
	}

	/** Sets the size of each bag sample, as a percentage of the training size. */
	public void setBagSizePercent(int bagSizePercent) {
	    BagSizePercent = bagSizePercent;
	}

	public double getSamplingPercentage() {
	    return samplingPercentage;
	}

	public void setSamplingPercentage(double samplingPercentage) {
	    this.samplingPercentage = samplingPercentage;
	}
	
	public int getEnsembleSize() {
		return ensembleSize;
	}
	
	public void setEnsembleSize(int ensembleSize) {
		this.ensembleSize = ensembleSize;
	}
	
}