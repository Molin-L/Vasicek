function [r,times]=simulateVasicekTbondPmeasure(  )
%SIMULATEVASICEK Generate simulations of the Vasickek model
rng('default')
r0=0.02;
a=0.5;
b=0.05;
sigma=0.05;
T=0.5;
lambda=-1;
NSteps=101;
dt=T/(NSteps-1);
times = 0:dt:T;
NPaths=100;
r = zeros( NPaths, NSteps );
currR = r0;
for i=1:NSteps
  epsilon = randn( NPaths,1 ); 
  r(:,i)=currR;
    currR = currR + (a.*(b-currR)-(lambda*sigma/a)).*dt + sigma.*sqrt(dt).*epsilon;
end
mean_P=mean(r, 1)
x=0:1:100
plot(x, mean_P)
end
   