
path = '/Users/xunan/Projects/MaxEnt/Data/MultiTE_DirAvg/REDIM_diravg.nii.gz';

signal = niftiread(path);


%%

te = ones(6,1)*[0.071 0.101 0.131 0.161 0.191];
b = (0:0.7:3.5)'* ones(1,5);

te_norm = ones(5,1)*[0.071 0.101 0.131 0.161 0.191];
b_norm = (0.7:0.7:3.5)'* ones(1,5);

qs = [b(:), te(:)];

%%  MaxEnt 

ND = 1000; NR = 1000;

D_ME = linspace(0.001,3,ND);
dD = gradient(D_ME); dD = dD/sum(dD)*3; 

R_ME = linspace(3,100,NR);
dR = gradient(R_ME); dR = dR/sum(dR)*97;

DD_ME = repmat(D_ME,[1 NR]);
RR_ME = kron(R_ME,ones(1,ND));

%%

spec_ME.D = D_ME;
spec_ME.R = R_ME;
spec_ME.Basis = exp(-te_norm(:)*RR_ME - b_norm(:)*DD_ME); %exp(-<q,theta>): M \times Q 
spec_ME.dTheta = kron(dR,dD); %1\times ! 
spec_ME.maxIter = 2000;
spec_ME.mu = 0.01^2; % I roughly checked the "MaxEntPDF_General" function and guess mu is the variance

%%

signal_norm = signal(:,:,:,qs(:,1) ~= 0) ./ repmat(signal(:,:,:,qs(:,1) == 0), [1, 1, 1, 5]);

qs_norm = qs(qs(:,1) ~= 0, :);

Sn = squeeze(signal_norm(44, 38, 43, :));


%%

[p,lambda,s_est]= MaxEntPDF_General(Sn,spec_ME);
p = reshape(p,ND,NR);

% figure;imagesc(D_ME(1:400),R_ME(1:400),p(1:400,1:400));
figure;imagesc(D_ME, R_ME, p);
axis xy; pbaspect([1 1 1]); colorbar; colormap('hot');

xlabel('D (\mum^2/ms)');
ylabel('R_2 (1/s)');
title('MaxEnt');




