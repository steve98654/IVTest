#include <cmath>
#include <boost/python.hpp>
#include <boost/math/distributions/normal.hpp>
using boost::math::normal;

// Implementation based on:
// https://mpra.ub.uni-muenchen.de/6867/1/MPRA_paper_6867.pdf
// Test case taken from Table 1 on page 47 (third parameter computed from eqn (1)
// Test hello_ext.MingOpt(0.6, -0.5, 0.238422, 1e-4)

// Constant used in many functions
const double OneOverRootTwoPi = 0.398942280401433;

// PDF of standard normal random variable
double NormalDensity(double x)
{
    return OneOverRootTwoPi*exp(-x*x/2);
}

// CDF of standard normal random variable
double CumulativeNormal(double x)
{
    static double a[5] = { 0.319381530,
                           -0.356563782,
                            1.781477937,
                           -1.821255978,
                            1.330274429};
    double result;
    if (x<-7.0)
        result = NormalDensity(x)/sqrt(1.+x*x);
    else
    {
        if (x>7.0)
            result = 1.0 - CumulativeNormal(-x);
        else
        {
            double tmp = 1.0/(1.0+0.2316419*fabs(x));
            result=1-NormalDensity(x)*
                (tmp*(a[0]+tmp*(a[1]+tmp*(a[2]+
                tmp*(a[3]+tmp*a[4])))));
            if (x<=0.0)
                result=1.0-result;
        }
    }
    return result;
}

// Inverse CDF of standard normal random variable
// Beasley-Springer/Moro approximation
double InverseCumulativeNormal(double u)
{
    static double a[4]={ 2.50662823884,
                        -18.61500062529,
                        41.39119773534,
                        -25.44106049637};
    static double b[4]={-8.47351093090,
                        23.08336743743,
                        -21.06224101826,
                        3.13082909833};
    static double c[9]={0.3374754822726147,
                        0.9761690190917186,
                        0.1607979714918209,
                        0.0276438810333863,
                        0.0038405729373609,
                        0.0003951896511919,
                        0.0000321767881768,
                        0.0000002888167364,
                        0.0000003960315187};
    double x=u-0.5;
    double r;

    if (fabs(x)<0.42) // Beasley-Springer
    {
        double y=x*x;
        r=x*(((a[3]*y+a[2])*y+a[1])*y+a[0])/
        ((((b[3]*y+b[2])*y+b[1])*y+b[0])*y+1.0);
    }
    else // Moro
    {
        r=u;
        if (x>0.0)
            r=1.0-u;
        r=log(-log(r));
        r=c[0]+r*(c[1]+r*(c[2]+r*(c[3]+r*(c[4]+r*(c[5]+
        r*(c[6]+r*(c[7]+r*c[8])))))));
        if (x<0.0)
        r=-r;
    }
    return r;
}

//v is the time scaled volatility, x is the moneyness, cstar is the observed
// option price, w is an optimization parameter
double MingG(double v, double x, double cstar, double w)
{
    double Nplus  = CumulativeNormal(x/v + 0.5*v);
    double Nminus = exp(-x)*CumulativeNormal(x/v - 0.5*v);
    double F = cstar + Nminus + w*Nplus;
    double tmp = InverseCumulativeNormal(F/(1+w));
    return tmp + sqrt(tmp*tmp + 2*fabs(x));
}

double Phi(double v, double x)
{
    return (v*v-2*fabs(x))/(v*v+2*fabs(x));
}

// SQR-DR, Equation (30) 
// Input initial guess for the time scaled implied vol v0, moneyness x, option market price cstar, and
// convergence tolerance tol of implied vol
double MingOpt(double v0, double x, double cstar, double tol)
{
    double w = Phi(v0,x);
    double v1 = v0;
    double v2 = MingG(v0,x, cstar, w);
    int cnt = 0;
    int max_it = 1000;

    while(fabs(v2-v1) > tol || cnt > max_it)
    {
        v1 = v2;
        v2 = MingG(v1,x,cstar,w);
        w = Phi(v2,x);
        cnt++;
    }

    return v2;

}
 
BOOST_PYTHON_MODULE(implied_vol_Li)
{
    using namespace boost::python;
    def("MingOpt", MingOpt);
}

