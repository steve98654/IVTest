#include <cmath>
#include <boost/python.hpp>
#include <boost/math/distributions/normal.hpp>
using boost::math::normal;

// https://mpra.ub.uni-muenchen.de/6867/1/MPRA_paper_6867.pdf
// Test hello_ext.MingOpt(0.6, -0.5, 0.238422, 1e-4)

const double OneOverRootTwoPi = 0.398942280401433;

double NormalDensity(double x)
{
    return OneOverRootTwoPi*exp(-x*x/2);
}

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

// the InverseCumulativeNormal function via the
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

// cstar is the observed option price
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

double BlackScholesCall( double Spot, double Strike, double r, double Vol, double Expiry)
{
    double standardDeviation = Vol*sqrt(Expiry);
    double moneyness = log(Spot/Strike);
    double d1=( moneyness + r*Expiry +
    0.5* standardDeviation*standardDeviation)/standardDeviation;
    double d2 = d1- standardDeviation;
    return Spot * CumulativeNormal(d1) - Strike*exp(-r*Expiry)*CumulativeNormal(d2);
}

double OwenOptPriceTest(double F, double K, double vol, double t, double df, int opt_type)
{
    double vol_root_t = vol*sqrt(t);
    double d1 = (log(F/K)+0.5*vol_root_t*vol_root_t)/vol_root_t;
    double d2 = d1 - vol_root_t;
    double px = opt_type*F*CumulativeNormal(opt_type*d1)-opt_type*K*CumulativeNormal(opt_type*d2);
    return px*df;
}

double OwenOptPriceTest2(double F, double K, double vol, double t, double df, int opt_type)
{
    normal s;
    double vol_root_t = vol*sqrt(t);
    double d1 = (log(F/K)+0.5*vol_root_t*vol_root_t)/vol_root_t;
    double d2 = d1 - vol_root_t;
    double px = opt_type*F*pdf(s,opt_type*d1)-opt_type*K*pdf(s,opt_type*d2);
    return px*df;
}

char const* greet()
{
   return "hello, world";
}
 
BOOST_PYTHON_MODULE(hello_ext)
{
    using namespace boost::python;
    def("greet", greet);
    def("BlackScholesCall", BlackScholesCall);
    def("OwenOptPriceTest", OwenOptPriceTest);
    def("OwenOptPriceTest2", OwenOptPriceTest2);
    def("MingOpt", MingOpt);
}
