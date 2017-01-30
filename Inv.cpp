#include <limits>

#define INF std::numeric_limits<float>::max()

class Inv
{
public:
    void viterbi(CSArrayF& activs, CSArrayI& labellings,
    int T, int N, int S, int min_skip, int max_skip, SArrayI& attention)
    {
        int M = max_skip + 1;
        if(M > T - S)
        {
            M = T - S + 1;
            if(M < 2)
            {
               M = 2;
            }
        }
        if(min_skip > M - 2)
        {
            min_skip = M - 2;
        }

        fwd_.resize(N * S, T + M - 1);
        bt_.resize(N * S, T + M - 1);
        score_.resize(N * S, T + M - 1);

        for(int t=0; t < T + M - 1; ++t)
            for(int s=0; s < N * S; ++s)
            {
                score_(s,t) = fwd_(s,t) = INF;
                bt_(s,t) = 1;
            }

        for(int t=0; t < T; ++t)
            for(int s=0; s < N * S; ++s)
                score_(s,t+M-1) = activs(t, labellings(s / S));

        for(int m = M-1 + min_skip; m < 2*M-2; ++m)
        {
            fwd_(0, m) = score_(0, m);
            bt_(0, m) = m - M + 2;
        }

        for(int s=1; s < N * S; ++s)
        {
            int start = T - (N * S - s) * M;
            if(start < 0)
                start = 0;
            //start = 0;
            for(int t=start; t < T; ++t)
            {
                float score = score_(s, t + M - 1);
                float min_score = INF;
                int min_index = M - min_skip;
                for(int m=t; m < t + M - min_skip; ++m)
                {
                    float prev = fwd_(s - 1, m);
                    if(prev < min_score)
                    {
                        min_score = prev;
                        min_index = m - t;
                    }
                }

                if(min_score == INF)
                  fwd_(s, t + M - 1) = INF;
                else
                  fwd_(s, t + M - 1) = min_score + score;
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
