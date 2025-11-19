
path = '/Users/xunan/Projects/MaxEnt/Data/MultiTE_DirAvg/REDIM_diravg.nii.gz';

signal = flip(niftiread(path), 1); % flip it such that the plot is the same as MRview   



%%

te = ones(6,1)*[0.071 0.101 0.131 0.161 0.191];
b = (0:0.7:3.5)'* ones(1,5);

te_norm = ones(5,1)*[0.071 0.101 0.131 0.161 0.191];
b_norm = (0.7:0.7:3.5)'* ones(1,5);

qs = [b(:), te(:)];

%%  MaxEnt 
%   looks like this is setting up integration 

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
spec_ME.mu = 0.01;

%%

signal_norm = signal(:,:,:,qs(:,1) ~= 0) ./ repmat(signal(:,:,:,qs(:,1) == 0), [1, 1, 1, 5]);

qs_norm = qs(qs(:,1) ~= 0, :);

Sn = squeeze(signal_norm(45, 66, 23, :));

S_square = signal_norm(31:45 , 56:70, 24, :);

imshow(imrotate(fliplr(signal(31:45 , 56:70, 24, 1)), -90), []);
colormap(gca, gray);


%%

% p_square = cell(15, 15);
% for i = 1:15
%    for j = 1:15
%         [temp_p,temp_lambda,temp_s_est]= MaxEntPDF_General(squeeze(S_square(i, j, :)),spec_ME);
%         temp_p = reshape(temp_p,ND,NR);
%         p_square{i, j} = temp_p;
%    end
% end

%%

% figure;
% t = tiledlayout(15, 15, 'Padding', 'none', 'TileSpacing', 'none');
% 
% idx = 0;
% for j = 15:-1:1             % columns: right to left
%     for i = 15:-1:1         % rows: bottom to top
%         idx = idx + 1;      % original indexing of your data
%         nexttile(t, idx);
%         imagesc(R_ME(1:10), D_ME(1:10), p_square{i, j});
%         axis off;
%     end
% end
% 
% colormap(hot)  % apply one global colormap

%%

[p,lambda,s_est]= MaxEntPDF_General(Sn,spec_ME);
p = reshape(p,ND,NR);

figure;imagesc(R_ME(1:400),D_ME(1:400),p(1:400,1:400));
% figure;imagesc(D_ME(:),R_ME(:),p(:,:) ./ (sum(sum(p)) * mean(dD) * mean(dR)) ); 
axis xy; pbaspect([1 1 1]); colorbar; colormap('hot');

xlabel('D (\mum^2/ms)');
ylabel('R_2 (1/s)');
title('MaxEnt');

%%
disp(sum(sum(p)) * mean(dD) * mean(dR));



