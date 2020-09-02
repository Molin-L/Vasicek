function [r,times]=simulateVasicekTbondQmeasure(  )
rng('default')
r0=0.02;
a=0.5;
b=0.05;
sigma=0.05;
T=0.5;
N=101;
dt = T/(N-1);
times = 0:dt:T;
NPaths=100;
r = zeros( NPaths, N );
currR = r0;
for i= 1:N
    epsilon = randn( NPaths,1 ); 
    r(:,i)=currR;
    currR = currR + a*(b-currR)*dt + sigma*sqrt(dt)*epsilon;

end 
end 

   