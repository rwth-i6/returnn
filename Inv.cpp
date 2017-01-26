#include <limits>

class Inv
{
public:
    void viterbi(CSArrayF& activs, CSArrayI& labellings,
    int T, int N, int S, int min_skip, int max_skip, SArrayI& attention)
    {
        int M = max_skip;
        if(M > T - S)
        {
            M = T - S;
            if(M < 1)
            {
               M = 1;
            }
        }

        fwd_.resize(N * S, T + max_skip - 1);
        bt_.resize(N * S, T + max_skip - 1);
        score_.resize(N * S, T + max_skip - 1);

        for(int t = 0; t < T + max_skip - 1; ++t)
            for(int s=0; s < N * S; ++s)
                score_(s,t) = fwd_(s,t) = std::numeric_limits<float>::max();

        for(int t = 0; t < T; ++t)
            for(int s=0; s < N * S; ++s)
                score_(s,t+M-1) = activs(t, labellings(s / S));

        for(int m = M-1; m < 2*M-2; ++m)
        {
            fwd_(0, m) = score_(0, m);
            bt_(0, m) = m - M + 2;
        }

        for(int s=1; s < N * S; ++s)
        {
            int start = T - (N * S - s) * M;
            if(start < 0)
                start = 0;
            for(int t=start; t < T; ++t)
            {
                float score = score_(s, t + M - 1);
                float min_score = std::numeric_limits<float>::max();
                float min_index = 0;
                for(int m=t; m < t + M; ++m)
                {
                    float prev = fwd_(s - 1, m);
                    if(prev + score < min_score)
                    {
                        min_score = prev + score;
                        min_index = m - t;
                    }
                }

                fwd_(s, t + M - 1) = min_score;
                bt_(s, t + M - 1) = M - 1 - min_index;
            }
        }

        int t = T - 1;
        attention(N*S-1) = T - 1;
        for(int s=N*S-2;s>=0;--s)
        {
            int next = t - bt_(s+1, t+M-1);
            attention(s) = next;
            t = next;
        }
    }
private:
    TwoDArray<float> fwd_;
    TwoDArray<float> score_;
    TwoDArray<int> bt_;
};
