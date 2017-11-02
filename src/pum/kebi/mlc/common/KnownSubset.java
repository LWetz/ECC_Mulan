package pum.kebi.mlc.common;

import java.io.Serializable;
import java.util.Arrays;

public class KnownSubset implements Serializable {
	
	private static final long serialVersionUID = -1101945247148745295L;
	public int numOccurances = 0;
	public boolean[] subset = null;
	@Override
	public int hashCode() {
		return Arrays.hashCode(subset);
	}
	@Override
	public boolean equals(Object obj) {
		return this.hashCode() == obj.hashCode();
	}
}