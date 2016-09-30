
class BestPathDecoder
{
public:
	void labellingErrors(CArrayF& activs, CArrayI& seqLengths, int idx, CArrayI& labellings, ArrayI& lev)
	{
		nLabels_ = activs.dim(2);
		T_ = seqLengths(idx);
		int len = calcLen(labellings, idx);

	    std::vector<int> labelling = label(activs, idx);
	    std::vector<int> reference(len);
	    for(int i = 0; i < len; ++i)
	    {
	        reference[i] = labellings(idx, i);
	    }

	    int dist = levenshteinDist(labelling, reference);
	    lev(idx) = dist;
	}
private:
	std::vector<int> label(CArrayF& activs, int idx)
	{
	    std::vector<int> labelling;
	    int lastLabel = ((nLabels_ + (nLabels_ % 2) - 1) / 2;
	    for(int t = 0; t < T_; ++t)
	    {
	        int bestLabel = 0;
	        float bestLabelProb = 0;
	        for(int l = 0; l < nLabels_; ++l)
	        {
	            float prob = activs(t, idx, l);
	            if(prob > bestLabelProb)
	            {
	                bestLabel = l / 2;
	                bestLabelProb = prob;
	            }
	        }
	        if(bestLabel != lastLabel)
	        {
	            labelling.push_back(bestLabel);
	            lastLabel = bestLabel;
	        }
	    }
	    return labelling;
	}

	int calcLen(CArrayI& labellings, int idx)
    {
        int len = labellings.dim(1);
        for(int j = 0; j < len; ++j)
        {
            if(labellings(idx, j) == -1)
            {
                return j;
            }
        }
        return len;
    }

    int T_;
    int nLabels_;
};
