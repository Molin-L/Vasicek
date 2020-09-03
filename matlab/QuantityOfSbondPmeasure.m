function [Qt]= QuantityOfSbondPmeasure()

Qt=DeltaforTBondPmeasure()./DeltaforSBondPmeasure();
mean_Qt=mean(Qt, 1)
x=0:1:100
plot(x, mean_Qt)
end