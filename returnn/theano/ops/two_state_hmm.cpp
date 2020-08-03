
class TwoStateHMM
{
public:
    void forwardBackward(CSArrayF& activs, CSArrayI& labellings,
        int seqLen, float& err, SArrayF& errSigs, SArrayF& priors, float tdp_loop, float tdp_fwd)
    {                
        nLabels_ = activs.dim(1);
        verify((nLabels_ + 1) % 2 == 0);
        si_ = (nLabels_ + 1) / 2 - 1;
        T_ = seqLen;
        N_ = calcLen(labellings);
        canonical_ = calcCanonical(labellings, N_);
        M_ = canonical_.size();
        errorSignal(activs, labellings, err, errSigs, priors, tdp_loop, tdp_fwd);
        skip_ = false;
    }
private:
    int emissionLabel(int chr, int state)
    {
      verify(state == 0 || state == 1, "state should be 0 or 1");
      return 2 * chr + state;
    }

    void forwardAlgo(CSArrayF& activs, CSArrayI& labellings, float tdp_loop, float tdp_fwd)
    {
        fwdTable_.resize(T_, M_);
        fwdTable_(0,0) = activs(0, canonical_[0]);

        for(int t = 1; t < T_; ++t)
        {
            for(int n = 0; n < M_; ++n)
            {
                myLog y(activs(t, canonical_[n]));
                int start = 0;
                if(skip_ && canonical_[n] % 2 == 0 && canonical_[n] != si_ && n > 1)
                {
                    start = n-2;
                }
                else if(n > 0)
                {
                    start = n-1;
                }
                myLog sum = 0;
                for(int i = start; i <= n; ++i)
                {
                    float tdp = 0;
                    if(i == n)
                    {
                      tdp = tdp_loop;
                    }
                    else
                    {
                      tdp = tdp_fwd;
                    }
                    sum += fwdTable_(t-1,i) * myLog(tdp);
                }
                fwdTable_(t,n) = y * sum;
            }
        }
    }

    void backwardAlgo(CSArrayF& activs, CSArrayI& labellings, float tdp_loop, float tdp_fwd)
    {
        bwdTable_.resize(T_, M_);
        bwdTable_(T_-1,M_-1) = 1;

        for(int t = T_-2; t != -1; --t)
        {
            for(int n = 0; n < M_; ++n)
            {
                int end = 0;
                if(n == M_-1)
                {
                    end = M_-1;
                }
                else
                {
                    end = n+1;
                }

                if(skip_ && n % 2 == 0 && n < M_-2 && n != si_)
                {
                    end = n+2;
                }
                
                myLog sum = 0;
                for(int i = n; i <= end; ++i)
                {
                    float tdp = 0;
                    if(i == n)
                    {
                      tdp = tdp_loop;
                    }
                    else
                    {
                      tdp = tdp_fwd;
                    }

                    myLog y = activs(t+1, canonical_[i]);
                    sum += bwdTable_(t+1,i) * y * myLog(tdp);
                }
                bwdTable_(t,n) = sum;
            }
        }
    }

    void errorSignal(CSArrayF& activs, CSArrayI& labellings, float& err, SArrayF& errSigs, SArrayF& priors,
                     float tdp_loop, float tdp_fwd)
    {
        forwardAlgo(activs, labellings, tdp_loop, tdp_fwd);
        backwardAlgo(activs, labellings, tdp_loop, tdp_fwd);

        myLog totalSum = 0;
        for(int t = 0; t < T_; ++t)
        {
            std::vector<myLog> labelSum(nLabels_);
            totalSum = 0;

            for(int n = 0; n < M_; ++n)
            {
                myLog prod = fwdTable_(t,n) * bwdTable_(t,n);
                labelSum[canonical_[n]] += prod;
                totalSum += prod;
            }
            for(int c = 0; c < nLabels_; ++c)
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
                std::cout << "Warning, HMM error of " << err << ", probably output sequence is too short. This warning is only printed once, even if the problem occurs multiple times." << std::endl;
                printedWarning = true;
            }
            err = 0;
            for(int t = 0; t < T_; ++t)
            {
                for(int c = 0; c < nLabels_; ++c)
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
        for(int i = 0; i < len; ++i)
        {
            int chr = labellings(i);
            if(chr == si_)
            {
                //silence has only 1 state
                canonical.push_back(emissionLabel(chr, 0));
            }
            else
            {
                canonical.push_back(emissionLabel(chr, 0));
                canonical.push_back(emissionLabel(chr, 1));
            }
        }
        return canonical;
    }

    bool skip_;
    int T_;
    int N_;
    int M_;
    int nLabels_;
    int si_;
    std::vector<int> canonical_;
    TwoDArray<myLog> fwdTable_;
    TwoDArray<myLog> bwdTable_;
};
