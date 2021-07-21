%% Overview
% This is a script that illustrates positive pressure event prediction in
% simulated furnace data. This script allows for 3 predictive models;
% dynamic PCA, auto-encoders and one-dimensional convolutional
% auto-encoders.

load('data.mat');
X = data.X;
X(isnan(X(:,1)),:)=[];
Pg = data.P;
i0 = data.I.F0;
i1 = data.I.F1;
it = data.I.Ft;
t = data.t;
BBs = find([0;diff(i1)]==1);
BBe = find([0;diff(i1)]==-1);
nBB = size(BBs,1);
model_type = 'CAE';
%% Generate model and residuals
switch model_type
    case 'dPCA'
        residuals = generate_pca_residuals(X,it);
        w = 360;
    case 'AE'
        residuals = generate_ae_residuals(X,it);
        w = 360;
    case 'CAE'
        residuals = generate_cae_residuals(X,it);
        w = 210;
end
discriminant = 1./residuals;
[modelPerfs,ithresh95,ithresh0bb,ithresh0fa,thresh95,thresh0bb,thresh0fa] = getModelPerfs(discriminant,w,i1,i0);
plotModelPerfs(discriminant,w,t,i1,i0,thresh95,thresh0bb,thresh0fa,Pg)
summarized_perf = give_performance_params(modelPerfs,ithresh95,ithresh0bb,ithresh0fa,thresh95,thresh0bb,thresh0fa,discriminant,w,t,Pg,i0,i1)

function residuals = generate_pca_residuals(X,it)
l = 4;
Xl = lagmatrix(X,[0:l]);
Xt = Xl(it,:);
meanXt = mean(Xt,1);
stdXt  = std(Xt,1);
Zt = (Xt-meanXt)./stdXt;
C = (Zt')*Zt;
[V,D] = eig(C);
[D,idx] = sortrows(diag(D),'descend');
V = V(:,idx);
varret = 99.9;
d = find(cumsum(D/sum(D))*100>varret,1);
P = V(:,1:d);
Zl = (Xl-meanXt)./stdXt;
Zr = Zl*P*(P');
R  = sum((Zr-Zl).^2,2);
residuals = R;
end

function residuals = generate_ae_residuals(X,it)
l = 4;
Xl = lagmatrix(X,[0:l]);
Xt = Xl(it,:);
meanXt = mean(Xt,1);
stdXt  = std(Xt,1);
Zt = (Xt-meanXt)./stdXt;
C = (Zt')*Zt;
[V,D] = eig(C);
[D,idx] = sortrows(diag(D),'descend');
V = V(:,idx);
Zt = Zt*V;
Ztm = mean(Zt,1); Zts = std(Zt,1);
Zt = (Zt-Ztm)./Zts;
Zl = (Xl-meanXt)./stdXt;
Zl = Zl*V;
Zl = (Zl-Ztm)./Zts;
Zc = Zt+(2*rand(size(Zt,1),size(Zt,2))-1)*1;
Zt = Zt'; Zc = Zc'; Zl = Zl';
net = fitnet([size(Zt,1) 3 size(Zt,1)]);
net.trainFcn = 'traingdm';
net.divideFcn = 'dividetrain';
net.trainParam.epochs = 2000;
net.performFcn = 'mse';
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
% net.trainParam.showWindow = 1;
net.performParam.regularization = 1e-5;
net = train(net,Zc,Zt);
Zr = net(Zl);
R = sum((Zl-Zr).^2,1); R = R';
residuals = R;
end

function residuals = generate_cae_residuals(X,it)
Xt = X(it,:);
meanXt = mean(Xt,1);
stdXt  = std(Xt,1);
Zt = (Xt-meanXt)./stdXt;
Z  = (X-meanXt)./stdXt;
C = (Zt')*Zt;
[V,D] = eig(C);
[D,idx] = sortrows(diag(D),'descend');
V = V(:,idx);
Z = Z*V; Zt = Zt*V;
meanZt = mean(Zt,1); stdZt = std(Zt,1);
Zt = (Zt-meanZt)./stdZt;
Z = (Z-meanZt)./stdZt;
Z = Z+abs(min(min(Z(it,:))));
[cnet, R] = applyCNN(Z,it,'shortNarrow');
residuals = R;
end

function Xim = MTStoImage(X,lagDimension)
l = lagDimension;
n = size(X,1);
m = size(X,2);

X = lagmatrix(X,[0:l]);
X = reshape(reshape(X',m,n*(l+1)),m,l+1,n);
Xim = reshape(X,m,l+1,1,n);
end

function [cnet, R] = applyCNN(X,idxTarget,cnnType)
m = size(X,2);
switch cnnType
    case 'shortNarrow'
        l = 4;
        X = MTStoImage(X,l);
        Xt = X(:,:,1,idxTarget); nt = size(Xt,4);
        Xc = Xt + (2*rand(m,l+1,1,nt)-1)*1;
        layers = shortNarrowCNN;
    case 'shortWide'
        l = 4;
        X = MTStoImage(X,l);
        Xt = X(:,:,1,idxTarget); nt = size(Xt,4);
        Xc = Xt + (2*rand(m,l+1,1,nt)-1)*1;
        layers = shortWideCNN;
    case 'longNarrow'
        l = 6;
        X = MTStoImage(X,l);
        Xt = X(:,:,1,idxTarget); nt = size(Xt,4);
        Xc = Xt + (2*rand(m,l+1,1,nt)-1)*1;
        layers = longNarrowCNN;
    case 'longWide'
        l = 6;
        X = MTStoImage(X,l);
        Xt = X(:,:,1,idxTarget); nt = size(Xt,4);
        Xc = Xt + (2*rand(m,l+1,1,nt)-1)*1;
        layers = longWideCNN;
end
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'MaxEpochs',100, ...
    'L2Regularization',1e-2,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.5,...
    'Plots','none'); %'none'

cnet = trainNetwork(Xc,Xt,layers,options);
Ri = sse(X,predict(cnet,X));
R = double(Ri(:));
end

function layers = shortNarrowCNN
layers = [...
    imageInputLayer([9,5,1])
    batchNormalizationLayer
    convolution2dLayer([1,3],8)
    reluLayer
    convolution2dLayer([1,3],8)
    reluLayer
    convolution2dLayer([9,1],4)
    reluLayer
    transposedConv2dLayer([9,5],1)
    regressionLayer];
end

function [modelPerfs,ithresh95,ithresh0bb,ithresh0fa,thresh95,thresh0bb,thresh0fa] = getModelPerfs(d,w,i1,i0)
    r = sum(binocdf([0:w],w,1)<0.9);
    nThresh = 200;
    f = (repelem(d,1,nThresh)-prctile(d,linspace(0,100,nThresh)))>=0;
    g = sum((movsum(f,[w,0],1)-r)>0,2)/2;
    h = g;
    threshes = [0:0.5:100]; Cr = size(threshes,2);
    bestData = zeros(Cr,12);
    for counti = 1:Cr
    thresh = threshes(counti);
    TP = sum(h(i1)>=thresh);
    FP = sum(h(i0)>=thresh);
    FN = sum(h(i1)<thresh);
    TN = sum(h(i0)<thresh);

    prec = TP/(TP+FP); bestData(counti,1) = prec;
    spec = TN/(TN+FP); bestData(counti,2) = spec;
    sens = TP/(TP+FN); bestData(counti,3) = sens;


    % Count missed, DD
    BBs = find([0;diff(i1)]==1);
    BBe = find([0;diff(i1)]==-1);
    nBB = size(BBs,1);
    nMiss = 0;
    DDidx  = zeros(nBB,1); DDidx(1:end) = nan;

    for j = 1:nBB
        q = zeros(size(i1,1),1);
        q(BBs(j)) = 1;
        q(BBe(j)) = -1;
        q = cumsum(q);
        qi = find(((h>thresh)+q)==2,1);
        if isempty(qi)==0
        DDidx(j) = qi;
        end
        if isempty(qi)==1
            DDidx(j) = BBe(j);
            nMiss = nMiss+1;
        end
    end
    DD = DDidx-BBs; DD = DD/360;
    DDmin = min(DD); DDmax = max(DD); DDmean = mean(DD);
    bestData(counti,4) = DDmin; bestData(counti,5) = DDmax; bestData(counti,6) = DDmean;
    tBB = BBe-DDidx; tBB = tBB/360+0.5;
    tBBmin = min(tBB); tBBmax = max(tBB); tBBmean = mean(tBB);
    bestData(counti,7) = tBBmin; bestData(counti,8) = tBBmax; bestData(counti,9) = tBBmean;
    mBB = sum(isnan(DD)); mBB = nMiss;
    MAR = mBB/nBB;
    bestData(counti,10) = mBB; bestData(counti,11) = MAR;
    bestData(counti,12) = FP;
    counti
    end
    modelPerfs = bestData;
    % Minimum specificity:
    ithresh95 = find(modelPerfs(:,1)>0.95,1);
    thresh95  = threshes(ithresh95);
    % No blowbacks:
    ithresh0bb = find(modelPerfs(:,10)>0,1)-1;
    thresh0bb  = threshes(ithresh0bb);
    % No false alarms:
    ithresh0fa = find(modelPerfs(:,2)==1,1);
    thresh0fa  = threshes(ithresh0fa);
end

function summarized_perf = give_performance_params(modelPerfs,ithresh95,ithresh0bb,ithresh0fa,thresh95,thresh0bb,thresh0fa,d,w,t,Pg,i0,i1)
    r = sum(binocdf([0:w],w,1)<0.9);
    nThresh = 200;
    f = (repelem(d,1,nThresh)-prctile(d,linspace(0,100,nThresh)))>=0;
    g = sum((movsum(f,[w,0],1)-r)>0,2)/2;
    h = g;
    summarized_perf.discPercentiles = h;
    summarized_perf.disc = d;
    summarized_perf.threshes.prec95 = thresh95;
    summarized_perf.threshes.bb0 = thresh0bb;
    summarized_perf.threshes.fa0 = thresh0fa;
    summarized_perf.t = t;
    summarized_perf.Pg = Pg;
    summarized_perf.i0 = i0;
    summarized_perf.i1 = i1;
    summarized_perf.prec95.dd = modelPerfs(ithresh95,6); % Detection delay, 7.03 h
    summarized_perf.prec95.wt = modelPerfs(ithresh95,9); % Warning time, 1.1 h
    summarized_perf.prec95.mbb = modelPerfs(ithresh95,10); % Missed blowbacks, 52
    summarized_perf.prec95.sens = modelPerfs(ithresh95,3); % Sensitivity, 3.71 %
    summarized_perf.prec95.spec = modelPerfs(ithresh95,2); % Specificity, 99.93 %
    % No missed BBs
    summarized_perf.bb0.dd = modelPerfs(ithresh0bb,6); % Detection delay, 3.03 h
    summarized_perf.bb0.wt = modelPerfs(ithresh0bb,9); % Warning time, 5.15 h
    summarized_perf.bb0.mbb = modelPerfs(ithresh0bb,10); % Missed blowbacks, 0
    summarized_perf.bb0.sens = modelPerfs(ithresh0bb,3); % Sensitivity, 52.04 %
    summarized_perf.bb0.spec = modelPerfs(ithresh0bb,2); % Specificity, 7.64 %
    % No false alarms
    summarized_perf.fa0.dd = modelPerfs(ithresh0fa,6); % Detection delay, 7.64 h
    summarized_perf.fa0.wt = modelPerfs(ithresh0fa,9); % Warning time, 53.51 %
    summarized_perf.fa0.mbb = modelPerfs(ithresh0fa,10); % Missed blowbacks, 62
    summarized_perf.fa0.sens = modelPerfs(ithresh0fa,3); % Sensitivity, 1.11 %
    summarized_perf.fa0.spec = modelPerfs(ithresh0fa,2); % Specificity, 100.00 %
end

function plotModelPerfs(d,w,t,i1,i0,thresh95,thresh0bb,thresh0fa,Pg)
    r = sum(binocdf([0:w],w,1)<0.9);
    nThresh = 200;
    f = (repelem(d,1,nThresh)-prctile(d,linspace(0,100,nThresh)))>=0;
    g = sum((movsum(f,[w,0],1)-r)>0,2)/2;
    tj = t/3600/24;
    yyaxis right
    plot(tj,g,'-b');
    yline(thresh95,'-.r','LineWidth',1);
    yline(thresh0bb,'-.g','LineWidth',1);
    yline(thresh0fa,'-.m','LineWidth',1);
    pbaspect([2,1,1]);
    a = gca;
    a.YColor = 'k';
    a.YLabel.String = 'Discriminant percentile';
    a.YLim = [0,200];
    a.YTick = [0:20:100];
    f = gcf;
    f.Color = [1,1,1];
    str1 = sprintf('Discriminant value');
    str2 = sprintf('Recognition threshold\nfor 95%% precision');
    str3 = sprintf('Recognition threshold\nfor no missed blowbacks');
    str4 = sprintf('Recognition threshold\nfor no false alarms');
    l = legend(str1,str2,str3,str4);
    % l = legend(str1,str3,str4);
    l.Location = 'bestoutside';
    l.AutoUpdate = 'off';

    yyaxis left
    plot(tj,Pg,'-k')
    ylimu = 6; yliml = -12;
    a = gca;
    tstart = 56;
    tdur = 9;
    a.XLim = [tstart,tstart+tdur];
    a.XLabel.String = 'Days of simulated operation';
    a.YLim = [yliml,ylimu];
    a.YLabel.String = 'Freeboard pressure [Pa]';
    a.XTick = [tstart:(tstart+tdur)];
    a.YColor = 'k';
    % i0 = idxNormal;
    % i1 = idxFault;
    f1start = find([0;diff(i1)]==1);
    f1end = find([0;diff(i1)]==-1);
    f0start = find([0;diff(i0)]==1);
    f0end = find([0;diff(i0)]==-1);
    n1 = size(f1end,1);
    n0 = size(f0end,1);
    hold on
    for i = 1:n1
        xi = tj(f1start(i));
        xj = tj(f1end(i));
        p = patch([xi,xj,xj,xi],[yliml,yliml,ylimu,ylimu],'r','FaceAlpha',0.05);
    end
    for i = 1:n0
        xi = tj(f0start(i));
        xj = tj(f0end(i));
        p = patch([xi,xj,xj,xi],[yliml,yliml,ylimu,ylimu],'g','FaceAlpha',0.05);
    end
    hold off
    pbaspect([1.6,1,1]);

end