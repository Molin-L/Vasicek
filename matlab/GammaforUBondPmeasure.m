function [gamma]=GammaforUBondPmeasure()
a=0.5;
b=0.05;
U=2;
NSteps=400;
du = U/NSteps;
u = 0:du:0.5;
sigma=0.05;
lambda=-1;
B=1/a.*(1-exp(-a.*(U-u)));
A=exp(((b-(sigma^2)/(2*a^2)-(lambda*sigma/a))*(B-U+u))-(((sigma^2)/(4*a)).*(B.^2)));
gamma=((-B).^2).*A.*exp(-B.*simulateVasicekTbondPmeasure( ));
end