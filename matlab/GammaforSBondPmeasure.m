function [gamma]=GammaforSBondPmeasure()
a=0.5;
b=0.05;
S=1;
NSteps=200;
ds = S/NSteps;
s = 0:ds:0.5;
sigma=0.05;
lambda=-1;
B=1/a.*(1-exp(-a.*(S-s)));
A=exp(((b-(sigma^2)/(2*a^2)-(lambda*sigma/a))*(B-S+s))-(((sigma^2)/(4*a)).*(B.^2)));
gamma=((-B).^2).*A.*exp(-B.*simulateVasicekTbondPmeasure( ));
mean_P=mean(gamma, 1)
x=0:1:100
plot(x, mean_P)
end