function[Z]=QuantityZPmeasure()
S=PriceforSbondPmeasure();
Qt=QuantityOfSbondPmeasure();
Zcb=zcbPricePmeasure();
MoneyReceived=zeros(100,101);
Z=zeros(100,101);
for i=2:101
MoneyReceived(:,i)=(Qt(:,i-1)-Qt(:,i)).*S(:,i);    
Z(:,1)=QuantityZat0Pmeasure();  
Z(:,i)=(MoneyReceived(:,i)+Z(:,i-1))./Zcb(:,i);
end
end

