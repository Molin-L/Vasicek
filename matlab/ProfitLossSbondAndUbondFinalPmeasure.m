function[PL]=ProfitLossSbondAndUbondFinalPmeasure()

T=PriceforTbondPmeasure();
Tbond=T(:,end);
Qt=QuantityOfSbondInUandSPmeasure();
QT=Qt(:,end);
Sbond=PriceforSbondPmeasure();
S=Sbond(:,end);
Qu= QuantityOfUbondInUandSPmeasure();
QU=Qu(:,end);
Ubond=PriceforUbondPmeasure();
U=Ubond(:,end);
Qz=QuantityZtInUandSPmeasure();
QZ=Qz(:,end);
Zcb=zcbPricePmeasure();
ZCB=Zcb(:,end);

PL=-Tbond+(QU.*U)+(QT.*S)+(QZ.*ZCB);

end
