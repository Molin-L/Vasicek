function[P]=PriceforTbond()
a=0.5;
b=0.05;
T=0.5;
NSteps=100;
dt = T/(NSteps);
t = 0:dt:0.5;
sigma=0.05;
B=1/a.*(1-exp(-a.*(T-t)));
A=exp(((b-(sigma^2)/(2*a^2))*(B-T+t))-(((sigma^2)/(4*a)).*(B.^2)));
P=A.*exp(-B.*simulateVasicekTbondQmeasure( ));
end