#include <limits>
#include <math.h>

#define INF std::numeric_limits<float>::max()
#define FOCUS_LAST 0
#define FOCUS_MAX 1


class Inv
{
public:
    void viterbi(CSArrayF& activs, CSArrayI& labellings,
    int T, int N, int S, int min_skip, int max_skip, int focus, SArrayI& attention)
    {
        int M = max_skip + 1;
        if(M > T - S)
        {
            M = T - S + 1;
            if(M < 1)
            {
               M = 1;
            }
        }
        if(min_skip > M)
        {
            min_skip = M;
        }
        if((T - M) / (N * S) > M)
        {
            M = (T - M) / (N * S) + 1;
            static int max_skip_warning_limit = 0;
            if(M > max_skip_warning_limit)
            {
                max_skip_warning_limit = M;
                cout << "warning: increasing max skip to " << M << " in order to avoid empty alignment" << endl;
            }
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
        for(int s=N*S-2;s>=-1;--s)
        {
            int next = t - bt_(s+1, t+M-1);
            if(s < 0)
                next = 0;
            if(focus == FOCUS_LAST)
                attention(s+1) = t;
            else if(focus == FOCUS_MAX)
            {
                float min_score = INF;
                int min_index = t;
                int upper = T - 1;
                if(s < N*S-2)
                    upper = t + bt_(s+2, t+M-1);
                upper = t;
                //cout << upper << "--" << T-1<<endl;
                for(int u=upper;u>next;--u)
                {
                    for(int c=0;c<N;++c)
                    {
                        if(min_score > activs(u,c))
                        {
                            min_score = activs(u,c);
                            min_index = u;
                        }
                    }
                }
                attention(s+1) = min_index;
            }
            t = next;
        }
    }

    void full(CSArrayF& activs, CSArrayI& labellings,
    int T, int N, int S, int min_skip, int max_skip, int focus, SArrayF& attention)
    {
        int M = max_skip + 1;
        if(M > T - S)
        {
            M = T - S + 1;
            if(M < 1)
            {
               M = 1;
            }
        }
        if(min_skip > M)
        {
            min_skip = M;
        }
        if(T / (N * S) > M)
        {
            M = T / (N * S) + 1;
            static int max_skip_warning_limit = 0;
            if(M > max_skip_warning_limit)
            {
                max_skip_warning_limit = M;
                cout << "warning: increasing max skip to " << M << " in order to avoid empty alignment" << endl;
            }
        }

        fwd_.resize(N * S, T + M - 1);
        bwd_.resize(N * S, T + M - 1);
        score_.resize(N * S, T + M - 1);

        for(int t=0; t < T + M - 1; ++t)
            for(int s=0; s < N * S; ++s)
            {
                score_(s,t) = fwd_(s,t) = bwd_(s,t) = INF;
            }

        for(int t=0; t < T; ++t)
            for(int s=0; s < N * S; ++s)
                score_(s,t+M-1) = activs(t, labellings(s / S));

        for(int m = M-1; m < 2*M-2; ++m)
        {
            fwd_(0, m) = score_(0, m);
        }

    	for(int s=1; s < N * S; ++s)
        {
            int start = T - (N * S - s) * M;
            if(start < 0)
                start = 0;
            start = 0;
            for(int t=start; t < T; ++t)
            {
                //float score = exp(-score_(s, t + M - 1));
                float sum = 0.0;
                for(int m=t; m < t + M; ++m)
                    sum += exp(-fwd_(s - 1, m + M - 1));
                fwd_(s, t + M - 1) = -log(sum) - score_(s, t + M - 1);
            }
        }

        bwd_(N * S - 1, T - 1) = 1.0; //score_(N * S - 1, T - 1);

        for(int s=N*S-2;s>=0; --s)
        {
            int start = T - (N * S - s) * M;
            if(start < 0)
                start = 0;
            start = 0;
            for(int t=start; t < T; ++t)
            {
                float score = exp(-score_(s, t));
                float sum = 0.0;
                for(int m=t; m < t + M; ++m)
                    sum += exp(-bwd_(s + 1, m));
                bwd_(s, t) = score_(s, t) - log(sum);
                /*
                float sum = 0;
                for(int i = s; i <= end; ++i)
                {
                    myLog y = score_(s, t + 1)
                    sum += bwdTable_(t+1,i) + y;
                }
                bwdTable_(t,n) = sum;
                */
            }
        }

        for(int s=0;s < N * S;++s)
        {
            for(int t=0; t < T; ++t)
            {
                attention(s, t) = exp(-fwd_(s, t)); // + exp(-bwd_(s, t)));
            }
        }
        /*
        for(int s=0;s < N * S;++s)
        {
            int chr = labellings(s/S);
            for(int t=0;t<T;++t)
            {
                attention(chr, t) += exp(-(fwd_(s, t) + bwd_(s, t)));
            }
        }
        */

        for(int s=0;s < N * S;++s)
            for(int t=0;t<T;++t)
                cout << labellings(s/S) << " " << t << " " << attention(labellings(s/S), t) << endl;
    }

private:
    TwoDArray<float> fwd_;
    TwoDArray<float> bwd_;
    TwoDArray<float> score_;
    TwoDArray<int> bt_;
};
