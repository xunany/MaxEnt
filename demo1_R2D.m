%two dimensional R2-D distribution

te = ones(6,1)*[0.07 0.1 0.13 0.16 0.19];
b = (0:1:5)'*ones(1,5);

SNR = 1000;
% model parameters
x = [18  .7 1;
     13  .3 1];%R2, D, vol-frac

S = zeros(size(b));
for i = 1:size(x,1)
    S = S + x(i,3)*exp(-x(i,1)*te-x(i,2)*b);
end

noise_level = mean(abs(S(:)))/SNR;

noise = noise_level*randn(size(S));
Sn = S + noise;
Sn = Sn/norm(Sn(:))*1e4;


%%  MaxEnt 

%looks like this is setting up integration 
ND = 1000; NR = 1000;
D_ME = linspace(0.001,3,ND);
dD = gradient(D_ME); dD = dD/sum(dD)*3; 
R_ME = linspace(3,100,NR);
dR = gradient(R_ME); dR = dR/sum(dR)*97;
DD_ME = repmat(D_ME,[1 NR]);
RR_ME = kron(R_ME,ones(1,ND));

spec_ME.D = D_ME;
spec_ME.R = R_ME;
spec_ME.Basis = exp(-te(:)*RR_ME-b(:)*DD_ME); %exp(-<q,theta>): M \times Q 
spec_ME.dTheta = kron(dR,dD); %1\times ! 
spec_ME.maxIter = 2000;
spec_ME.mu = 1e-4;

[p,lambda,s_est]= MaxEntPDF_General(Sn,spec_ME);
p = reshape(p,ND,NR);
% figure;imagesc(R_ME(1:400),D_ME(1:400),p(1:400,1:400));
figure;imagesc(R_ME,D_ME,p);
xlabel('R_2 (1/s)');
ylabel('D (\mum^2/ms)');
title('MaxEnt');
