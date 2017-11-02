package pum.kebi.mlc.common;
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
 *    MacroAUC.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */


import mulan.evaluation.measure.BipartitionMeasureBase;
import mulan.evaluation.measure.MacroAverageMeasure;

/**
 * Implementation of the macro-averaged Accuracy measure.
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.10
 */
public class MacroAccuracy extends BipartitionMeasureBase implements MacroAverageMeasure {

	private static final long serialVersionUID = -5368786502744335861L;
	
	private int numOfLabels = 0;
	private int[] tps = null;
	private int[] tns = null;
	private int[] fps = null;
	private int[] fns = null;
	
    /**
     * Creates a new instance of this class
     *
     * @param numOfLabels the number of labels
     */
    public MacroAccuracy(int numOfLabels) {
        this.numOfLabels = numOfLabels;
        this.tps = new int[numOfLabels];
        this.tns = new int[numOfLabels];
        this.fps = new int[numOfLabels];
        this.fns = new int[numOfLabels];
    }

    public String getName() {
        return "Macro-averaged Accuracy";
    }

    public double getValue() {
    	double sum = 0d;
    	for (int l = 0; l < numOfLabels; l++) {
    		sum += getValue(l);
		}
    	return sum / (double)numOfLabels; 
    }

    /**
     * Returns the accuracy for a particular label
     * 
     * @param labelIndex the index of the label 
     * @return the AUC for that label
     */
    public double getValue(int l) {
        return ((double)(tps[l]+tns[l]))/(double)(tps[l]+tns[l]+fps[l]+fns[l]);  
    }

	@Override
	public double getIdealValue() {
		return 1d;
	}

	@Override
	public void reset() {
		this.tps = new int[numOfLabels];
        this.tns = new int[numOfLabels];
        this.fps = new int[numOfLabels];
        this.fns = new int[numOfLabels];
	}

	@Override
	protected void updateBipartition(boolean[] bipartition, boolean[] truth) {
		
		for(int l = 0; l < this.numOfLabels; l++) {
			this.tps[l] += truth[l] && bipartition[l] ? 1 : 0;
			this.fps[l] += truth[l] && !bipartition[l] ? 1 : 0;
			this.tns[l] += !truth[l] && !bipartition[l] ? 1 : 0;
			this.fns[l] += !truth[l] && bipartition[l] ? 1 : 0;
		}
		
	}
	
	
	

}