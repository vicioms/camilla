#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
using namespace std;

int pmod(int a, int b)
{
    return (a % b + b) % b;
}

double sum(const double* array, int size)
{
    double total = 0.0;
    for (int i = 0; i < size; ++i)
        total += array[i];
    return total;
}

int sample_event(
    const double* rates,
    double total_rate,
    double u_rand,
    int num_events
)
{
    double remaining = u_rand * total_rate;

    for (int i = 0; i < num_events; ++i)
    {
        remaining -= rates[i];

        if (remaining < 0.0)
            return i;
    }

    // Numerical roundoff fallback
    return num_events - 1;
}

struct state1d
{
    int L;
    int N;

    vector<int> s_plus;
    vector<int> s_minus;
    vector<int> i_plus;
    vector<int> i_minus;

    state1d(int l, int n)
        : L(l),
          N(n),
          s_plus(l, 0),
          s_minus(l, 0),
          i_plus(l, 0),
          i_minus(l, 0)
    {}
};


class Epidemic1d
{
public:
    state1d x;

    int N_s = 0;
    int N_i = 0;

    double event_rates[5];

    mt19937_64 generator;
    uniform_real_distribution<double> distribution;



    Epidemic1d(int L, int N)
        : x(L, N),
          generator(random_device{}()),
          distribution(0.0, 1.0)
    {}

    void init_random()
    {
        for (int n = 0; n < x.N; ++n)
        {
            int site = static_cast<int>(
                distribution(generator) * x.L
            );

            if (distribution(generator) < 0.5)
                x.s_plus[site]++;
            else
                x.s_minus[site]++;
        }

        N_s = x.N;
        N_i = 0;
    };

    void infect_random(int num_infected)
    {
        num_infected = std::min(num_infected, N_s);

        for (int n = 0; n < num_infected; ++n)
        {
            double remaining = distribution(generator) * N_s;

            for (int i = 0; i < x.L; ++i)
            {
                if (remaining < x.s_plus[i])
                {
                    x.s_plus[i]--;
                    x.i_plus[i]++;
                    N_s--;
                    N_i++;
                    break;
                }
                remaining -= x.s_plus[i];

                if (remaining < x.s_minus[i])
                {
                    x.s_minus[i]--;
                    x.i_minus[i]++;
                    N_s--;
                    N_i++;
                    break;
                }
                remaining -= x.s_minus[i];
            }
        }
    }

    // Select one particle uniformly and hop
    int hop(vector<int>& plus,
            vector<int>& minus,
            int total)
    {
        if (total == 0)
            return -1;

        double remaining =
            distribution(generator) * total;

        for (int i = 0; i < x.L; ++i)
        {
            int total_i = plus[i] + minus[i];

            remaining -= total_i;

            if (remaining < 0.0)
            {
                // We already know the selected particle
                // is somewhere at site i.
                double r =
                    distribution(generator) * total_i;

                if (r < plus[i])
                {
                    plus[i]--;
                    plus[pmod(i + 1, x.L)]++;
                }
                else
                {
                    minus[i]--;
                    minus[pmod(i - 1, x.L)]++;
                }

                return i;
            }
        }

        return -1;
    }

    // Select one particle among the entire population
    // and reverse its orientation.
    void tumble()
    {
        int total = N_s + N_i;

        if (total == 0)
            return;

        double remaining =
            distribution(generator) * total;

        for (int i = 0; i < x.L; ++i)
        {
            int counts[4] = {
                x.s_plus[i],
                x.s_minus[i],
                x.i_plus[i],
                x.i_minus[i]
            };

            for (int type = 0; type < 4; ++type)
            {
                remaining -= counts[type];

                if (remaining < 0.0)
                {
                    switch (type)
                    {
                        case 0:
                            x.s_plus[i]--;
                            x.s_minus[i]++;
                            return;

                        case 1:
                            x.s_minus[i]--;
                            x.s_plus[i]++;
                            return;

                        case 2:
                            x.i_plus[i]--;
                            x.i_minus[i]++;
                            return;

                        case 3:
                            x.i_minus[i]--;
                            x.i_plus[i]++;
                            return;
                    }
                }
            }
        }
    }

    double compute_infection_weight()
    {
        double weight = 0.0;

        for (int i = 0; i < x.L; ++i)
        {
            int S = x.s_plus[i] + x.s_minus[i];
            int I = x.i_plus[i] + x.i_minus[i];

            weight += static_cast<double>(S) * I;
        }

        return weight;
    }

    void infect()
    {
        double total_weight = compute_infection_weight();

        if (total_weight <= 0.0)
            return;

        double remaining =
            distribution(generator) * total_weight;

        for (int i = 0; i < x.L; ++i)
        {
            int S = x.s_plus[i] + x.s_minus[i];
            int I = x.i_plus[i] + x.i_minus[i];

            double w =
                static_cast<double>(S) * I;

            remaining -= w;

            if (remaining < 0.0)
            {
                // Preserve orientation of susceptible
                double r =
                    distribution(generator) * S;

                if (r < x.s_plus[i])
                {
                    x.s_plus[i]--;
                    x.i_plus[i]++;
                }
                else
                {
                    x.s_minus[i]--;
                    x.i_minus[i]++;
                }

                N_s--;
                N_i++;

                return;
            }
        }
    }

    void recover()
    {
        if (N_i == 0)
            return;

        double remaining =
            distribution(generator) * N_i;

        for (int i = 0; i < x.L; ++i)
        {
            int I = x.i_plus[i] + x.i_minus[i];

            remaining -= I;

            if (remaining < 0.0)
            {
                double r =
                    distribution(generator) * I;

                if (r < x.i_plus[i])
                {
                    x.i_plus[i]--;
                    x.s_plus[i]++;
                }
                else
                {
                    x.i_minus[i]--;
                    x.s_minus[i]++;
                }

                N_i--;
                N_s++;

                return;
            }
        }
    }

    double step(
        double tumbling_rate,
        double hopping_rate_s,
        double hopping_rate_i,
        double infection_rate,
        double recovery_rate
    )
    {
        double infection_weight =
            compute_infection_weight();

        event_rates[0] =
            tumbling_rate * (N_s + N_i);

        event_rates[1] =
            hopping_rate_s * N_s;

        event_rates[2] =
            hopping_rate_i * N_i;

        event_rates[3] =
            infection_rate * infection_weight;

        event_rates[4] =
            recovery_rate * N_i;

        double total_rate =
            sum(event_rates, 5);

        if (total_rate <= 0.0)
            return INFINITY;

        int event = sample_event(
            event_rates,
            total_rate,
            distribution(generator),
            5
        );

        // Gillespie waiting time
        exponential_distribution<double>
            waiting_time(total_rate);

        double dt = waiting_time(generator);

        switch (event)
        {
            case 0:
                tumble();
                break;

            case 1:
                hop(
                    x.s_plus,
                    x.s_minus,
                    N_s
                );
                break;

            case 2:
                hop(
                    x.i_plus,
                    x.i_minus,
                    N_i
                );
                break;

            case 3:
                infect();
                break;

            case 4:
                recover();
                break;
        }

        return dt;
    }

    int susceptible() const
    {
        return N_s;
    }

    int infected() const
    {
        return N_i;
    }
};


int main()
{
    int L = 1024;
    int N = 512;
    double tumbling_rate = 1.0;
    double hopping_rate_s = 1.0;
    double hopping_rate_i = 0.1;
    double infection_rate = 0.01;
    double recovery_rate = 0.001;
    Epidemic1d model(L, N);
    model.init_random();
    model.infect_random(int(N * 0.1));

    ofstream outfile("epidemic_output.dat", ios::out | ios::binary);

    for (int sweep = 0; sweep < 10000; ++sweep)
    {
        for (int step = 0; step < N; ++step)
        {
            model.step(
                tumbling_rate,
                hopping_rate_s,
                hopping_rate_i,
                infection_rate,
                recovery_rate
            );
        };
        
        outfile.write(reinterpret_cast<const char*>(model.x.s_plus.data()), model.x.s_plus.size() * sizeof(int));
        outfile.write(reinterpret_cast<const char*>(model.x.s_minus.data()), model.x.s_minus.size() * sizeof(int));
        outfile.write(reinterpret_cast<const char*>(model.x.i_plus.data()), model.x.i_plus.size() * sizeof(int));
        outfile.write(reinterpret_cast<const char*>(model.x.i_minus.data()), model.x.i_minus.size() * sizeof(int));
    }
    return 0;
}