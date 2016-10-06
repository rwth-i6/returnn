
class BestPathDecoder
{
public:
	void labellingErrors(CArrayF& activs, CArrayI& seqLengths, int idx, CArrayI& labellings, ArrayI& lev)
	{
		nLabels_ = activs.dim(2);
		T_ = seqLengths(idx);
		int len = calcLen(labellings, idx);
		si_ = nLabels_ - 1;

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
	    int lastLabel = nLabels_ - 1;
	    for(int t = 0; t < T_; ++t)
	    {
	        int bestLabel = 0;
	        float bestLabelProb = 0;
	        if(lastLabel % 2 == 1 || lastLabel == si_)
	        {
              for(int l = 0; l < nLabels_; l += 2)
              {
                  float prob = activs(t, idx, l);
                  if(prob > bestLabelProb)
                  {
                      bestLabel = l;
                      bestLabelProb = prob;
                  }
              }
             }
            else
            {
                 float loop = activs(t, idx, lastLabel);
                 float forward = activs(t, idx, lastLabel+1);
                 bestLabel = lastLabel + (int)(forward > loop);
            }
	        if(bestLabel != lastLabel)
	        {
	            if (lastLabel % 2 == 1)
	              labelling.push_back(lastLabel/2);
	            lastLabel = bestLabel;
	        }
	        else if(t == T_ -1 && bestLabel % 2 == 1)
	        {
	            labelling.push_back(bestLabel/2);
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
    int si_;
    int nLabels_;
};
