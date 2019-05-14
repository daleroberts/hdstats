#include "randomkit.h"

static double rk_normal(rk_state *state, double loc, double scale)
{
    return loc + scale*rk_gauss(state);
}

static double rk_standard_exponential(rk_state *state)
{
    /* We use -log(1-U) since U is [0, 1) */
    return -log(1.0 - rk_double(state));
}

static double rk_exponential(rk_state *state, double scale)
{
    return scale * rk_standard_exponential(state);
}

static double rk_uniform(rk_state *state, double loc, double scale)
{
    return loc + scale*rk_double(state);
}

static double rk_standard_gamma(rk_state *state, double shape)
{
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0)
    {
        return rk_standard_exponential(state);
    }
    else if (shape < 1.0)
    {
        for (;;)
        {
            U = rk_double(state);
            V = rk_standard_exponential(state);
            if (U <= 1.0 - shape)
            {
                X = pow(U, 1./shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -log((1-U)/shape);
                X = pow(1.0 - shape + shape*Y, 1./shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1./3.;
        c = 1./sqrt(9*b);
        for (;;)
        {
            do
            {
                X = rk_gauss(state);
                V = 1.0 + c*X;
            } while (V <= 0.0);

            V = V*V*V;
            U = rk_double(state);
            if (U < 1.0 - 0.0331*(X*X)*(X*X)) return (b*V);
            if (log(U) < 0.5*X*X + b*(1. - V + log(V))) return (b*V);
        }
    }
}

static double rk_gamma(rk_state *state, double shape, double scale)
{
    return scale * rk_standard_gamma(state, shape);
}

static double rk_chisquare(rk_state *state, double df)
{
    return 2.0*rk_standard_gamma(state, df/2.0);
}

static double rk_noncentral_chisquare(rk_state *state, double df, double nonc)
{
    if (nonc == 0){
        return rk_chisquare(state, df);
    }
    if(1 < df)
    {
        const double Chi2 = rk_chisquare(state, df - 1);
        const double N = rk_gauss(state) + sqrt(nonc);
        return Chi2 + N*N;
    }
    else
    {
        const long i = rk_poisson(state, nonc / 2.0);
        return rk_chisquare(state, df + 2 * i);
    }
}

static double rk_f(rk_state *state, double dfnum, double dfden)
{
    return ((rk_chisquare(state, dfnum) * dfden) /
            (rk_chisquare(state, dfden) * dfnum));
}

static double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc)
{
    double t = rk_noncentral_chisquare(state, dfnum, nonc) * dfden;
    return t / (rk_chisquare(state, dfden) * dfnum);
}
