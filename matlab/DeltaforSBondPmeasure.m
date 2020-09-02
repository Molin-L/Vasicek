function [delta]=DeltaforSBondPmeasure()
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
delta=-B.*A.*exp(-B.*simulateVasicekTbondPmeasure( ));
end