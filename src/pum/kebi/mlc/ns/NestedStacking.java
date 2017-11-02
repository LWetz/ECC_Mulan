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
 *    ClassifierChain.java
 *    Copyright (C) 2012 Robin Senge, University of Marburg, Germany
 */
package pum.kebi.mlc.ns;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import pum.kebi.mlc.common.KnownSubset;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 <!-- globalinfo-start -->
 * Class implementing the Nested Stacking (NS) algorithm.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * TODO publish?
 * <p/>
 <!-- globalinfo-end -->
 *
 * @author Robin Senge [senge@mathematik.uni-marburg.de]
 */
public class NestedStacking extends TransformationBasedMultiLabelLearner {

	private static final long serialVersionUID = -4296859994493125025L;

	/** The new chain ordering of the label indices. */
	private int[] labelOrder;

	/** If the predictions along the chain are probabilities or classes. */
	private boolean propagateProbabilities = false;

	/** If true, only subsets of labels are predicted, which have occured in the training data before. */
	private boolean subsetCorrected = false;

	/** The way to predict the classes during training phase. */
	private NestedPredictionMethod nestedPredictionMethod = NestedPredictionMethod.TRAINING;
	
	
	/**
	 * The ensemble of binary relevance models. These are Weka
	 * FilteredClassifier objects, where the filter corresponds to removing all
	 * label apart from the one that serves as a target for the corresponding
	 * model.
	 */
	protected FilteredClassifier[] ensemble;
	
	/**
	 * A hashtable containing the known label subses.
	 */
	protected Hashtable<KnownSubset, KnownSubset> knownSubsets = null;
	
	/**
	 * The known label subsets in an array.
	 */
	protected KnownSubset[] knownSubsetArray = null;

	/**
	 * Creates a new NestedStacker with a Logistic base learner. 
	 */
	public NestedStacking() {
		this(new Logistic());
	}
	
	/**
	 * Creates a new object of this model.
	 */
	public NestedStacking(Classifier classifier) {
		super(classifier);
		if(nestedPredictionMethod != NestedPredictionMethod.TRAINING) 
			throw new IllegalArgumentException("Nested prediction method not implemented, yet: " + nestedPredictionMethod);
	}

	/**
	 * Returns a string describing the classifier.
	 */
	@Override
	public String globalInfo() {
		return "Class implementing the Nested Stacking (NS) algorithm." 
		+ "\n\n" + "For more information, see\n\n" 
		+ getTechnicalInformation().toString();
	}

	protected void buildInternal(MultiLabelInstances train) throws Exception {

		// standard order 1--k
		if (labelOrder == null) {
			labelOrder = new int[numLabels];
			for (int i = 0; i < numLabels; i++) {
				labelOrder[i] = i;
			}
		}

		Instances trainDataset;
		numLabels = train.getNumLabels();
		ensemble = new FilteredClassifier[numLabels];
		trainDataset = train.getDataSet();
		Instances[] variants = null; 
		
		if(subsetCorrected) {
			
			knownSubsets = new Hashtable<KnownSubset,KnownSubset>();
			int[] labels = train.getLabelIndices();
			double[][] labelColumns = new double[numLabels][];
			for (int l = 0; l < labelColumns.length; l++) {
				labelColumns[l] = train.getDataSet().attributeToDoubleArray(labels[l]);
			}
			Instances dataset = train.getDataSet();
			for(int i = 0; i < train.getNumInstances(); i++) {
				boolean[] tmp = new boolean[numLabels];
				for (int l = 0; l < labelColumns.length; l++) {
					tmp[l] = dataset.attribute(labels[l]).value((int)dataset.instance(i).value(labels[l])).equals("1");
				}
				
				KnownSubset tmpSet = new KnownSubset();
				tmpSet.subset = tmp;
				
				if(!knownSubsets.contains(tmpSet)) {
					knownSubsets.put(tmpSet, tmpSet);
					tmpSet.numOccurances = 1;
				} else {
					knownSubsets.get(tmpSet).numOccurances++;
				}
			}
			
			this.knownSubsetArray = new KnownSubset[this.knownSubsets.size()];
			Enumeration<KnownSubset> enumSubset = this.knownSubsets.keys();
			int i = 0;
			while (enumSubset.hasMoreElements()) {
				this.knownSubsetArray[i++] = enumSubset.nextElement();
			}
			
		}
		
		if(propagateProbabilities) { 
			
			List<Instance> raw = new ArrayList<Instance>(); 
			for(Instance inst : trainDataset) 
				raw.add(new DenseInstance(1d, inst.toDoubleArray()));
			
			// create one dataset variant for each label, which interprets the values
			// for the previous labels as numerical values (probabilities)
			variants = new Instances[numLabels];
			
			// change attribute type of labels to numeric, to be able to pass in probabilities
			for (int i = 0; i < numLabels; i++) {

				if(i == 0) {
					variants[i] = trainDataset;
					continue;
				} else {
					variants[i] = new Instances(variants[i-1], raw.size());
				}
				
				// add numeric attribute
				Attribute pAtt = new Attribute("p" + trainDataset.attribute(labelIndices[labelOrder[i-1]]).name());
				variants[i].insertAttributeAt(pAtt, labelIndices[labelOrder[i-1]] + 1);

				// delete the label attribute
				variants[i].deleteAttributeAt(labelIndices[labelOrder[i-1]]);
				
				for(Instance inst : raw) 
					variants[i].add(inst);
				
			}
			
		} 


		for (int i = 0; i < numLabels; i++) {

			ensemble[i] = new FilteredClassifier();
			ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

			// Indices of attributes to remove first removes numLabels attributes
			// the numLabels - 1 attributes and so on.
			// The loop starts from the last attribute.
			int[] indicesToRemove = new int[numLabels - 1 - i];
			int counter2 = 0;
			for (int counter1 = 0; counter1 < numLabels - i - 1; counter1++) {
				indicesToRemove[counter1] = labelIndices[labelOrder[numLabels - 1 - counter2]];
				counter2++;
			}
			
			Instances data = null;
			Instances nextData = null;

			if(propagateProbabilities) {
				data = variants[i];
				nextData = i < numLabels - 1 ? variants[i+1] : null;
			} else {
				data = trainDataset;
			}
			
			Remove remove = new Remove();
			remove.setAttributeIndicesArray(indicesToRemove);
			remove.setInputFormat(data);
			remove.setInvertSelection(false);
			ensemble[i].setFilter(remove);

			data.setClassIndex(labelIndices[labelOrder[i]]);
			debug("Bulding model " + (i + 1) + "/" + numLabels);
			ensemble[i].buildClassifier(data);

			
			debug("Replacing actual class values by predictions of label " + (i + 1) + "/" + numLabels);
			
			if(propagateProbabilities && nextData != null) {
				
				for(int j = 0; j < data.numInstances(); j++) {
					double[] dist = ensemble[i].distributionForInstance(data.instance(j));
					nextData.instance(j).setValue(
							labelIndices[labelOrder[i]], 
							dist[data.classAttribute().value(0).equals("1") ? 0 : 1]);
				}

			} else {

				for (Iterator<Instance> iterator = trainDataset.iterator(); iterator.hasNext();) {
					Instance instance = iterator.next();
					
					double[] dist = ensemble[i].distributionForInstance(instance);
					if(data.classAttribute().value(0).equals("1")) {
						instance.setClassValue((dist[0] > dist[1]) ? 1 : 0); // rounding: positive effect?
					} else {
						instance.setClassValue((dist[0] < dist[1]) ? 1 : 0); // rounding: positive effect?
					}

				}
			}	
		}
		
	}

	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
		
		boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];

		Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
		for (int l = 0; l < numLabels; l++) {
			double distribution[];
			try {
				distribution = ensemble[l].distributionForInstance(tempInstance);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = ensemble[l].getFilter().getOutputFormat().classAttribute();
			bipartition[labelOrder[l]] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[labelOrder[l]] = distribution[classAttribute.indexOfValue("1")];

			tempInstance.setValue(labelIndices[labelOrder[l]], propagateProbabilities ? confidences[labelOrder[l]] : (double) maxIndex);

		}
		
		if(subsetCorrected) {
			bipartition = correctSubset(bipartition);
		}

		MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
		return mlo;
		
	}
	
	
	private boolean[] correctSubset(final boolean[] subset) {
		
		Arrays.sort(knownSubsetArray, new Comparator<KnownSubset>() {

			@Override
			public int compare(KnownSubset o1, KnownSubset o2) {
				
				int numMiss1 = 0;
				int numMiss2 = 0;
				
				for(int i = 0; i < o1.subset.length; i++) {
					numMiss1 += subset[i] == o1.subset[i] ? 0 : 1;
					numMiss2 += subset[i] == o2.subset[i] ? 0 : 1;
				}

				if(numMiss1 == numMiss2) {
					// decide with relative frequencies
					return o1.numOccurances > o2.numOccurances ? -1 : +1;
				}

				if(numMiss1 < numMiss2) {
					return -1;
				} else {
					return +1;
				}
				
			}
		
		});
		
		return knownSubsetArray[0].subset.clone();
		
	}

	public static enum NestedPredictionMethod {
		
		/** train once using all training samples */
		TRAINING,

		/** do a 10-fold cross-validation */
		TEN_CV,

		/** do a loave-one-out cross-validation */
		LEAVE_ONE_OUT
	}
	
	public boolean isPropagateProbabilities() {
		return propagateProbabilities;
	}
	
	public void setPropagateProbabilities(boolean propagateProbabilities) {
		this.propagateProbabilities = propagateProbabilities;
	}
	
	public boolean isSubsetCorrected() {
		return subsetCorrected;
	}
	
	public void setSubsetCorrected(boolean subsetCorrected) {
		this.subsetCorrected = subsetCorrected;
	}
	
	public int[] getLabelOrder() {
		return labelOrder;
	}
	
	public void setLabelOrder(int[] labelOrder) {
		this.labelOrder = labelOrder;
	}


}