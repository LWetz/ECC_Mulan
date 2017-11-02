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
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package pum.kebi.mlc.cc;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Hashtable;

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
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 <!-- globalinfo-start -->
 * Class implementing the Classifier Chain (CC) algorithm.<br/>
 * <br/>
 * For more information, see<br/>
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
 * 
 <!-- technical-bibtex-end -->
 * <p/>
 * 
 * <p>
 * <strong>Added options by Senge:</strong>
 * <ul>
 * <li><i>propagate probabilities</i>: propagates the probability for the positive class 
 * instead of class predictions to the next classifier in the chain.</li>
 * <li><i>subset correction</i>: enables the post-processing step of subset correction 
 * that only predicts subsets that have been seen before (in the training data) </li>
 * </ul>
 * 
 * </p>
 * 
 *
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * @author Konstantinos Sechidis (sechidis@csd.auth.gr)
 * @author Grigorios Tsoumakas (greg@csd.auth.gr)
 * @author Robin Senge (senge@mathematik.uni-marburg.de)
 */
public class ClassifierChain extends TransformationBasedMultiLabelLearner {

	private static final long serialVersionUID = 95770191373989915L;

	/**
	 * The new chain ordering of the label indices
	 */
	private int[] chain;

	/**
	 * Propagate the probability for the positive class instead of class predictions to the next classifier in the chain.
	 */
	private boolean propagateProbabilities = false;


	/**
	 * If true, only subsets of labels are predicted, which have occurred in the training data before.
	 */
	private boolean subsetCorrected = false;


	/**
	 * Returns a string describing the classifier.
	 *
	 * @return a string description of the classifier 
	 */
	@Override
	public String globalInfo() {
		return "Class implementing the Classifier Chain (CC) algorithm." 
		+ "\n\n" + "For more information, see\n\n" 
		+ getTechnicalInformation().toString();
	}

	@Override
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

	/**
	 * The ensemble of binary relevance models. These are Weka
	 * FilteredClassifier objects, where the filter corresponds to removing all
	 * label apart from the one that serves as a target for the corresponding
	 * model.
	 */
	protected FilteredClassifier[] ensemble;


	/**
	 * A hashtable containing the known label subsets.
	 */
	protected Hashtable<KnownSubset, KnownSubset> knownSubsets = null;

	/**
	 * The known label subsets in an array.
	 */
	protected KnownSubset[] knownSubsetArray = null;

	/**
	 * Creates a new instance using Logistic as the underlying classifier.
	 */
	public ClassifierChain() {
		super(new Logistic());
	}

	/**
	 * Creates a new instance.
	 *
	 * @param classifier the base-level classification algorithm that will be
	 * used for training each of the binary models
	 */
	public ClassifierChain(Classifier classifier) {
		super(classifier);
	}
	

	protected void buildInternal(MultiLabelInstances train) throws Exception {

		if (chain == null) {
			chain = new int[numLabels];
			for (int i = 0; i < numLabels; i++) {
				chain[i] = i;
			}
		}

		Instances trainDataset;
		numLabels = train.getNumLabels();
		ensemble = new FilteredClassifier[chain.length];
		trainDataset = train.getDataSet();

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


		// added to enable classifier chains to deal with incomplete labels
		int[] sortedChain = Arrays.copyOf(chain, chain.length);
		Arrays.sort(sortedChain);
		int[] ignoreIndices = new int[numLabels-chain.length];
		int j = 0;
		for (int l = 0; l < numLabels; l++) {
			if(Arrays.binarySearch(sortedChain, l) < 0) {
				ignoreIndices[j++] = labelIndices[l];
			}
		}

		for (int i = 0; i < chain.length; i++) {

			ensemble[i] = new FilteredClassifier();
			ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

			// Indices of attributes to remove first removes numLabels attributes
			// the numLabels - 1 attributes and so on.
			// The loop starts from the last attribute.
			int[] labelsToRemove = new int[chain.length - 1 - i];
			int counter2 = 0;
			for (int counter1 = 0; counter1 < chain.length - i - 1; counter1++) {
				labelsToRemove[counter1] = labelIndices[chain[chain.length - 1 - counter2]];
				counter2++;
			}

			// remove ignored labels and labels, which occur later in the chain
			int[] indicesToRemove = new int[labelsToRemove.length + ignoreIndices.length];
			System.arraycopy(labelsToRemove, 0, indicesToRemove, 0, labelsToRemove.length);
			System.arraycopy(ignoreIndices, 0, indicesToRemove, labelsToRemove.length, ignoreIndices.length);

			Remove remove = new Remove();
			remove.setAttributeIndicesArray(indicesToRemove);
			remove.setInputFormat(trainDataset);
			remove.setInvertSelection(false);
			ensemble[i].setFilter(remove);

			trainDataset.setClassIndex(labelIndices[chain[i]]);
			debug("Bulding model " + (i + 1) + "/" + numLabels);
			ensemble[i].buildClassifier(trainDataset);
		}
	}

	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
		boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];
		Arrays.fill(confidences, Double.NaN);

		Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
		for (int counter = 0; counter < chain.length; counter++) {

			double distribution[];
			try {
				distribution = ensemble[counter].distributionForInstance(tempInstance);
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
			bipartition[chain[counter]] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[chain[counter]] = distribution[classAttribute.indexOfValue("1")];

			tempInstance.setValue(labelIndices[chain[counter]], this.propagateProbabilities ? confidences[chain[counter]] : (double) maxIndex);

		}

		// remove ignored labels
		/*boolean[] subbipartition = new boolean[chain.length];
        double[] subconfidences = new double[chain.length];
        int j = 0;
        for (int i = 0; i < confidences.length; i++) {
			if(!Double.isNaN(confidences[i])) {
				subconfidences[j] = confidences[i];
				subbipartition[j] = bipartition[i];
				j++;
			}
		}*/

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
	
	public int[] getChain() {
		return chain;
	}
	
	public void setChain(int[] chain) {
		this.chain = chain;
	}

	public boolean isSubsetCorrected() {
		return subsetCorrected;
	}
	
	public void setSubsetCorrected(boolean subsetCorrected) {
		this.subsetCorrected = subsetCorrected;
	}
	
	public boolean isPropagateProbabilities() {
		return propagateProbabilities;
	}
	
	public void setPropagateProbabilities(boolean propagateProbabilities) {
		this.propagateProbabilities = propagateProbabilities;
	}
	
	


}