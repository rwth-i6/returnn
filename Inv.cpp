#include <limits>
#include <math.h>

#define INF std::numeric_limits<float>::max()
#define FOCUS_LAST 0
#define FOCUS_MAX 1

#define COVERAGE_EXPONENTIAL 1
#define COVERAGE_UNIFORM 2
#define COVERAGE_CONSTANT 3
#define COVERAGE_DENSE 4

#define DEBUG 0
#define VERBOSE 1
#define AUTO_INCREASE_SKIP 1

class Inv
{
public:
    void viterbi(CSArrayF& activs, CSArrayI& labellings,
    int T, int N, int S, int min_skip, int max_skip, int focus, int nil, int coverage, SArrayF& attention)
    {
        int M = max_skip;
        if(AUTO_INCREASE_SKIP)
        {
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
          if(M == 0)
          {
            M = T / (N * S) + min_skip + 1;
            min_skip = M - 2 * min_skip;
          }
          else if((T - M) / (N * S) > M)
          {
              M = (T - M) / (N * S) + 1;
              static int max_skip_warning_limit = 0;
              if(VERBOSE && M > max_skip_warning_limit)
              {
                  max_skip_warning_limit = M;
                  cout << "warning: increasing max skip to " << M << " in order to avoid empty alignment" << endl;
              }
          }
        }

        fwd_.resize(N * S, T + M - 1);
        bt_.resize(N * S, T + M - 1);
        score_.resize(N * S, T + M - 1);

        for(int t=0; t < T + M - 1; ++t)
            for(int s=0; s < N * S; ++s)
            {
                score_(s,t) = fwd_(s,t) = INF;
                bt_(s,t) = min_skip;
            }

        for(int t=0; t < T; ++t)
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
            start = 0;
            start = s + 1;
            int cur_min_skip = min_skip;
            //if(labellings(s / S) == nil)
            //  cur_min_skip = 0;
            for(int t=start; t < T; ++t)
            {
                int cur_max_skip = M;
                //if(labellings(s / S) == nil)
                //  cur_max_skip = T - t;
                float score = score_(s, t + M - 1);
                float min_score = INF;
                int min_index = cur_max_skip;
                for(int m=t + min_skip; m < t + cur_max_skip - 1; ++m)
                {
                    float prev = fwd_(s - 1, m);
                    if(prev < min_score)
                    {
                        min_score = prev;
                        min_index = m - t;
                    }
                }

                //cerr << s << " " << t << " " << min_score << " " << min_index << endl;

                if(min_score == INF)
                {
                  fwd_(s, t + M - 1) = INF;
                  bt_(s, t + M - 1) = cur_min_skip;
                }
                else
                {
                  fwd_(s, t + M - 1) = min_score + score;
                  bt_(s, t + M - 1) = cur_max_skip - 1 - min_index;
                }
            }
        }

        int t = T - 1;
        for(int s=N*S-2;s>=-1;--s)
        {
            int next = t - bt_(s+1, t + M - 1);
            //cout << s+1 << ": " << t << " -> " << next << " (" << T << "," << N << ")" << endl;
            if(next > t)
            {
                cout << "warning: backward trace detected at " << s+1 << ": " << t << " -> " << next << endl;
            }
            if(next == t)
            {
                cout << "warning: loop in inverted alignment detected at " << s+1 << " " << t << endl;
            }
            if(t < 0)
            {
                cout << "warning: negative time index detected" << endl;
            }
            if(next < 0)
                next = -1;
            if(focus == FOCUS_LAST)
            {
                if(labellings((s+1) / S) == nil)
                {
                  for(int i=t;i>next;--i)
                    attention(s+1,i) = 1. /((float)(t-next));
                }
                else if(coverage > 0)
                {
                  for(int i=t;i>next;--i)
                  {
                    switch(coverage)
                    {
                        case COVERAGE_UNIFORM: attention(s+1,i) = 1./((float)(t-next));break;
                        case COVERAGE_EXPONENTIAL: attention(s+1,i) = 1./((float)(t-i+1));break;
                        case COVERAGE_CONSTANT: attention(s+1,i) = 1;break;
                        case COVERAGE_DENSE: attention(0,i) = labellings((s+1) / S);break;
                        default: cout << "unknown coverage flag: " << coverage << endl;break;
                    }
                  }
                }
                else
                {
                  if(attention(s+1, t) != 0)
                    cout << "warning: attention at " << s+1 << " " << t << " has value " << attention(s+1,t) << endl;
                  attention(s+1, t) = 1;
                  //attention(s + 1, t) = exp(-fwd_(s + 1, t + M - 1));
                  //cerr << attention(s + 1, t) << endl;
                }
            }
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
                attention(s+1, min_index) = 1;
            }
            t = next;
        }

        if(DEBUG)
        {
          for(int s=N*S-2;s>=-1;--s)
          {
              float sum = 0;
              for(int t=0;t<T;++t)
              {
                  sum += attention(s+1,t);
                  if(sum>1)
                  {
                      cout << "warning: multiple alignment points on single frame at " << s << " " << t << endl;
                      throw std::out_of_range("alignment error");
                      break;
                  }
              }
          }
        }
    }

    void viterbi_backtrace(CSArrayF& activs, CSArrayI& labellings,
    int T, int N, int S, int min_skip, int max_skip, int focus, int nil, int coverage, SArrayF& attention, SArrayI& bt)
    {
        int M = max_skip;
        if(AUTO_INCREASE_SKIP)
        {
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
          if(M == 0)
          {
            M = T / (N * S) + min_skip + 1;
            min_skip = M - 2 * min_skip;
          }
          else if((T - M) / (N * S) > M)
          {
              M = (T - M) / (N * S) + 1;
              static int max_skip_warning_limit = 0;
              if(VERBOSE && M > max_skip_warning_limit)
              {
                  max_skip_warning_limit = M;
                  cout << "warning: increasing max skip to " << M << " in order to avoid empty alignment" << endl;
              }
          }
        }

        fwd_.resize(N * S, T + M - 1);
        bt_.resize(N * S, T + M - 1);
        score_.resize(N * S, T + M - 1);

        for(int t=0; t < T + M - 1; ++t)
            for(int s=0; s < N * S; ++s)
            {
                score_(s,t) = fwd_(s,t) = INF;
                bt_(s,t) = min_skip;
            }

        for(int t=0; t < T; ++t)
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
            start = 0;
            start = s + 1;
            int cur_min_skip = min_skip;
            //if(labellings(s / S) == nil)
            //  cur_min_skip = 0;
            for(int t=start; t < T; ++t)
            {
                int cur_max_skip = M;
                //if(labellings(s / S) == nil)
                //  cur_max_skip = T - t;
                float score = score_(s, t + M - 1);
                float min_score = INF;
                int min_index = cur_max_skip;
                for(int m=t + min_skip; m < t + cur_max_skip - 1; ++m)
                {
                    float prev = fwd_(s - 1, m);
                    if(prev < min_score)
                    {
                        min_score = prev;
                        min_index = m - t;
                    }
                }

                //cerr << s << " " << t << " " << min_score << " " << min_index << endl;

                if(min_score == INF)
                {
                  fwd_(s, t + M - 1) = INF;
                  bt_(s, t + M - 1) = cur_min_skip;
                }
                else
                {
                  fwd_(s, t + M - 1) = min_score + score;
                  bt_(s, t + M - 1) = cur_max_skip - 1 - min_index;
                }
            }
        }

        int t = T - 1;
        for(int s=N*S-2;s>=-1;--s)
        {
            int next = t - bt_(s+1, t + M - 1);
            for(int tt=0;tt<T;++tt)
                bt(s+1, tt) = bt_(s+1, tt + M - 1);
            //cout << s+1 << ": " << t << " -> " << next << " (" << T << "," << N << ")" << endl;
            if(next > t)
            {
                cout << "warning: backward trace detected at " << s+1 << ": " << t << " -> " << next << endl;
            }
            if(next == t)
            {
                cout << "warning: loop in inverted alignment detected at " << s+1 << " " << t << endl;
            }
            if(t < 0)
            {
                cout << "warning: negative time index detected" << endl;
            }
            if(next < 0)
                next = -1;
            if(focus == FOCUS_LAST)
            {
                if(labellings((s+1) / S) == nil)
                {
                  for(int i=t;i>next;--i)
                    attention(s+1,i) = 1. /((float)(t-next));
                }
                else if(coverage > 0)
                {
                  for(int i=t;i>next;--i)
                  {
                    switch(coverage)
                    {
                        case COVERAGE_UNIFORM: attention(s+1,i) = 1./((float)(t-next));break;
                        case COVERAGE_EXPONENTIAL: attention(s+1,i) = 1./((float)(t-i+1));break;
                        case COVERAGE_CONSTANT: attention(s+1,i) = 1;break;
                        case COVERAGE_DENSE: attention(0,i) = labellings((s+1) / S);break;
                        default: cout << "unknown coverage flag: " << coverage << endl;break;
                    }
                  }
                }
                else
                {
                  if(attention(s+1, t) != 0)
                    cout << "warning: attention at " << s+1 << " " << t << " has value " << attention(s+1,t) << endl;
                  attention(s+1, t) = 1;
                  //attention(s + 1, t) = exp(-fwd_(s + 1, t + M - 1));
                  //cerr << attention(s + 1, t) << endl;
                }
            }
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
                attention(s+1, min_index) = 1;
            }
            t = next;
        }

        if(DEBUG)
        {
          for(int s=N*S-2;s>=-1;--s)
          {
              float sum = 0;
              for(int t=0;t<T;++t)
              {
                  sum += attention(s+1,t);
                  if(sum>1)
                  {
                      cout << "warning: multiple alignment points on single frame at " << s << " " << t << endl;
                      throw std::out_of_range("alignment error");
                      break;
                  }
              }
          }
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
            if(VERBOSE && M > max_skip_warning_limit)
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
                score_(s,t) = fwd_(s,t) = bwd_(s,t) = INF;

        for(int t=0; t < T; ++t)
            for(int s=0; s < N * S; ++s)
                score_(s,t+M-1) = activs(t, labellings(s / S));

        for(int m = M-1; m < 2*M-2; ++m)
        {
            fwd_(0, m) = score_(0, m);
            //bwd_(N * S - 1, T - 1 - m + M + 1) = score_(N*S-1, T - 1 - m + M + 1);
        }

      	for(int s=1; s < N * S; ++s)
        {
            int start = T - (N * S - s) * M;
            if(start < 0)
                start = 0;
            //start = 0;
            for(int t=start; t < T - (N * S - s - 1); ++t)
            {
                float sum = 0.0;
                for(int m=t; m < t + M; ++m)
                    sum += exp(-(score_(s, t + M - 1) + fwd_(s - 1, m)));
                fwd_(s, t + M - 1) = -log(sum);
            }
        }

        //bwd_(N * S - 1, T - 1) = 0.0; //score_(N * S - 1, T - 1);
        int offset = M - 1;
        bwd_(N * S - 1, T - 1) = score_(N * S - 1,offset + T - 1);

        for(int s=N*S-2; s>=0; --s)
        {
            int start = 0;
            //start = M - 1;
            for(int t=start; t < T; ++t)
            {
                float sum = 0.0;
                for(int m=t; m < t + M; ++m)
                  sum += exp(-(score_(s, t + offset)));
                bwd_(s, t + offset) = -log(sum);
            }
        }

        for(int s=0; s < N * S; ++s)
        {
          for(int t=0; t < T; ++t)
            fwd_(s, t) += bwd_(s, t + offset);
        }

        for(int t=0; t < T; ++t)
        {
          float sum = 0.0;
          for(int s=0; s < N * S; ++s)
            sum += exp(-fwd_(s,t));
          for(int s=0; s < N * S; ++s)
            attention(s,t) = exp(-fwd_(s,t)) + sum;
        }
    }

public:
    TwoDArray<float> fwd_;
    TwoDArray<float> bwd_;
    TwoDArray<float> score_;
    TwoDArray<int> bt_;
};



class Std
{
public:
    void viterbi(CSArrayF& activs, CSArrayI& labellings,
    int T, int N, int S, int skip_tdp, SArrayI& attention)
    {

    }

    void full(CSArrayF& activs, CSArrayI& labellings,
    int T, int N, int S, int skip_tdp, SArrayI& alignment)
    {
        fwd_.resize(T, N * S + 2);
        bwd_.resize(T, N * S + 2);
        score_.resize(T, N * S + 2);

        for(int s=0; s < N * S; ++s)
            for(int t=0; t < T + 2; ++t)
                score_(t,s) = fwd_(s,t) = bwd_(s,t) = INF;

        for(int s=0; s < N * S; ++s)
            for(int t=0; t < T; ++t)
                score_(t,s) = activs(t, labellings(s / S));

        fwd_(0,2) = score_(0,0);
        for(int t=1; t < T; ++t)
            for(int s=0; s < N * S; ++s)
            {
                float sum = 0.0;
                for(int m=0;m<3;++m)
                {
                    sum += exp(-(fwd_(t - 1, s + m) + score_(t,s)));
                }
                fwd_(t, s+2) = -log(sum);
            }

        bwd_(T - 1, N * S - 1) = score_(T - 1, N * S - 1);
        for(int t=T-2;t>=0;--t)
            for(int s=N * S - 1; s >= 0; --s)
            {
                float sum = 0.0;
                for(int m=0;m<3;++m)
                {
                    sum += exp(-(fwd_(t + 1, s + m) + score_(t,s)));
                }
                bwd_(t, s) = -log(sum);
            }

        for(int t=0; t < T; ++t)
        {
            float sum = 0.0;
            for(int s=0;s < N * S;++s)
            {
                if(fwd_(s, t) == INF || bwd_(s,t) == INF)
                    alignment(t, s) = 0;
                else
                    alignment(t, s) = exp(-(fwd_(t, s) + bwd_(t, s)));
                sum += alignment(t, s);
            }
            for(int s=0;s < N * S;++s)
                alignment(t, s) /= sum;
        }
    }

private:
    TwoDArray<float> fwd_;
    TwoDArray<float> bwd_;
    TwoDArray<float> score_;
    TwoDArray<int> bt_;
};



class InvAlign
{
public:
    void viterbi(CSArrayF& activs, CSArrayI& labellings,
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
                attention(s+1, t) = 1;
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
                attention(s+1, min_index) = 1;
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
                score_(s,t) = fwd_(s,t) = bwd_(s,t) = INF;

        for(int t=0; t < T; ++t)
            for(int s=0; s < N * S; ++s)
                score_(s,t+M-1) = activs(t, labellings(s / S));

        for(int m = M-1; m < 2*M-2; ++m)
        {
            fwd_(0, m) = score_(0, m);
            //bwd_(N * S - 1, T - 1 - m + M + 1) = score_(N*S-1, T - 1 - m + M + 1);
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
                    sum += exp(-fwd_(s - 1, m));
                //cout << "sum:" << sum << endl;
                if(sum > 0 && score_(s, t + M - 1) != INF)
                    fwd_(s, t + M - 1) = -log(sum) + score_(s, t + M - 1);
            }
        }

        //bwd_(N * S - 1, T - 1) = 0.0; //score_(N * S - 1, T - 1);
        bwd_(N * S - 1, T - 1) = score_(N * S - 1, T - 1);

        for(int s=N*S-2;s>=0; --s)
        {
            int start = T - (N * S - s) * M;
            if(start < 0)
                start = 0;
            start = 0;
            //start = M - 1;
            for(int t=start; t < T; ++t)
            {
                float sum = 0.0;
                for(int m=t; m < t + M; ++m)
                    if(bwd_(s + 1, m) != INF && score_(s, t) != INF)
                        sum += exp(-bwd_(s + 1, m) - score_(s, m));
                /*
                for(int m=t; m > t - M; ++m)
                    if(m >= 0 && bwd_(s + 1, m) != INF && score_(s, t) != INF)
                        sum += exp(-bwd_(s + 1, m) - score_(s+1, m));
                */
                if(sum > 0.0)
                    bwd_(s, t) = -log(sum);
            }
        }

        for(int s=0;s < N * S;++s)
        {
            float sum = 0.0;
            for(int t=0; t < T; ++t)
            {
                //bwd_(s, t) = 0;
                if(fwd_(s, t + M - 1) == INF || bwd_(s,t) == INF)
                    attention(s, t) = 0;
                else
                    attention(s, t) = exp(-(fwd_(s, t + M - 1) + bwd_(s, t)));
                //attention(s, t) = exp(-(fwd_(s, t + M - 1) + bwd_(s, t)));
                //cout << s << " " << t << " fw " << fwd_(s, t) << " bw " << bwd_(s, t) << endl;
                //attention(s, t) = exp(-(fwd_(s, t + M - 1) + bwd_(s, t)));
                //attention(s, t) = exp(-(fwd_(s, t + M - 1) + bwd_(s, t)));
                /*
                if(s < N*S-1)
                {
                    for(int m=t;m<t+M;++m)
                    {
                        attention(s, t) += bwd_(s+1,m);
                    }
                }*/
                //sum += exp(-(fwd_(s, t + M - 1) + bwd_(s, t)));
                sum += attention(s, t);
            }
            for(int t=0; t < T; ++t)
                attention(s, t) /= sum;
        }

        /*for(int s=0;s < N * S;++s)
            for(int t=0;t<T;++t)
                cout << labellings(s/S) << " " << t << " " << attention(s, t) << endl;
        */
    }

private:
    TwoDArray<float> fwd_;
    TwoDArray<float> bwd_;
    TwoDArray<float> score_;
    TwoDArray<int> bt_;
};
