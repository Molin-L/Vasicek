function[P]=PriceforTbondPmeasure()
a=0.5;
b=0.05;
T=0.5;
NSteps=101;
dt=T/(NSteps-1);
t=0:dt:T;
sigma=0.05;
lambda=-1;
B=1/a.*(1-exp(-a.*(T-t)));
A=exp(((b-(sigma^2)/(2*a^2)-(lambda*sigma/a))*(B-T+t))-(((sigma^2)/(4*a)).*(B.^2)));
P=A.*exp(-B.*simulateVasicekTbondPmeasure( ));
mean_P=mean(P, 1)
x=0:1:100
plot(x, mean_P)
end