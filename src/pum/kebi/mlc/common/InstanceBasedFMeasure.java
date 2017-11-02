package pum.kebi.mlc.common;
/*
 * Copyright (C) 2012, Poznan University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import mulan.evaluation.measure.ExampleBasedBipartitionMeasureBase;

/**
 * This class calculates example based F measure, correction of the
 * mulan.evaluation.measure.ExampleBasedFMeasure class.
 * @author Adrian Jaroszewicz
 */
public class InstanceBasedFMeasure extends ExampleBasedBipartitionMeasureBase {

	private static final long serialVersionUID = -267641156033420969L;
	private double beta = 1.0;

	@Override
    protected void updateBipartition(boolean[] prediction, boolean[] truth) {
		int sumPrediction = 0;
		int sumTruth = 0;
		int sumMul = 0;
		for (int i = 0; i < truth.length; i++) {
			if (prediction[i])
				sumPrediction++;
			if (truth[i])
				sumTruth++;
			int b = prediction[i]? 1 : 0;
			int t = truth[i]? 1 : 0;
			sumMul += b * t;
		}
		
		if (sumPrediction == 0 && sumTruth == 0)
			sum += 1;
		else
			sum += ((1 + beta * beta) * sumMul) / (sumPrediction + beta * beta * sumTruth);
		count++;
    }

	@Override
	public double getIdealValue() {
		return 1;
	}

	@Override
	public String getName() {
		return "My Example-Based F Measure";
	}
}