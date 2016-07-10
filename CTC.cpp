
class CTC
{
public:
    void forwardBackward(CSArrayF& activs, CSArrayI& labellings,
        int seqLen, float& err, SArrayF& errSigs, SArrayF& priors)
    {                
        nLabelsInclBlank_ = activs.dim(1);
        blankIdx_ = nLabelsInclBlank_ - 1;
        T_ = seqLen;
        N_ = calcLen(labellings);
        M_ = 2 * N_ + 1;
        canonical_ = calcCanonical(labellings, N_);            
        errorSignal(activs, labellings, err, errSigs, priors);
    }
private:
    void forwardAlgo(CSArrayF& activs, CSArrayI& labellings)
    {
        fwdTable_.resize(T_, M_);
        fwdTable_(0,0) = activs(0, nLabelsInclBlank_ - 1);
        fwdTable_(0,1) = activs(0, labellings(0));

        for(int t = 1; t < T_; ++t)
        {
            for(int n = 0; n < M_; ++n)
            {
                myLog y(activs(t, canonical_[n]));
                int start = 0;
                if(n == 0 || n == 1)
                {
                    start = 0;
                }
                else if(canonical_[n] == blankIdx_ || canonical_[n] == canonical_[n-2])
                {
                    start = n-1;
                }
                else
                {
                    start = n-2;
                }
                myLog sum = 0;
                for(int i = start; i <= n; ++i)
                {
                    sum += fwdTable_(t-1,i);
                }
                fwdTable_(t,n) = y * sum;
            }
        }
    }

    void backwardAlgo(CSArrayF& activs, CSArrayI& labellings)
    {
        bwdTable_.resize(T_, M_);
        bwdTable_(T_-1,M_-1) = 1;
        bwdTable_(T_-1,M_-2) = 1;

        for(int t = T_-2; t != -1; --t)
        {
            for(int n = 0; n < M_; ++n)
            {
                int end = 0;
                if(n == M_-1 || n == M_-2)
                {
                    end = M_-1;
                }
                else if(canonical_[n] == blankIdx_ 
                    || canonical_[n] == canonical_[n+2])
                {
                    end = n+1;
                }
                else
                {
                    end = n+2;
                }
                
                myLog sum = 0;
                for(int i = n; i <= end; ++i)
                {
                    myLog y = activs(t+1, canonical_[i]);
                    sum += bwdTable_(t+1,i) * y;
                }
                bwdTable_(t,n) = sum;
            }
        }
    }

    void errorSignal(CSArrayF& activs, CSArrayI& labellings, float& err, SArrayF& errSigs, SArrayF& priors)
    {
        forwardAlgo(activs, labellings);
        backwardAlgo(activs, labellings);

        myLog totalSum = 0;
        for(int t = 0; t < T_; ++t)
        {
            std::vector<myLog> labelSum(nLabelsInclBlank_);
            totalSum = 0;

            for(int n = 0; n < M_; ++n)
            {
                myLog prod = fwdTable_(t,n) * bwdTable_(t,n);
                labelSum[canonical_[n]] += prod;
                totalSum += prod;
            }
            for(int c = 0; c < nLabelsInclBlank_; ++c)
            {
                errSigs(t, c) = activs(t, c) - (labelSum[c] / totalSum).expVal();
                priors(c) += activs(t, c) - errSigs(t,c);
            }
        }
        err = -totalSum.logVal();
        
        if(err > 1e10)
        {
            static bool printedWarning = false;
            if(!printedWarning)
            {
                std::cout << "Warning, CTC error of " << err << ", probably output sequence is too short. This warning is only printed once, even if the problem occurs multiple times." << std::endl;
                printedWarning = true;
            }
            err = 0;
            for(int t = 0; t < T_; ++t)
            {
                for(int c = 0; c < nLabelsInclBlank_; ++c)
                {
                    errSigs(t, c) = 0;
                    priors(c) = 0;
                }
            }
        }
    }

    int calcLen(CSArrayI& labellings)
    {
        int len = labellings.dim(0);
        for(int j = 0; j < len; ++j)
        {
            if(labellings(j) == -1)
            {
                return j;
            }
        }
        return len;
    }

    std::vector<int> calcCanonical(CSArrayI& labellings, int len)
    {
        std::vector<int> canonical;
        canonical.push_back(blankIdx_);
        for(int i = 0; i < len; ++i)
        {
            canonical.push_back(labellings(i));
            canonical.push_back(blankIdx_);
        }
        return canonical;
    }

    int T_;
    int N_;
    int M_;
    int nLabelsInclBlank_;
    int blankIdx_;
    std::vector<int> canonical_;
    TwoDArray<myLog> fwdTable_;
    TwoDArray<myLog> bwdTable_;
};
