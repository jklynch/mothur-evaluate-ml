%
% combine shared and design files on the command line
% with mothur_files.py
%
function evaluate_lmnn()

%echo off;
%clear all;
%clc;
rand('seed',1);
setpaths
fprintf('Loading data ...\n');
d = load('/home/jklynch/gsoc2013/mothur-evaluate-ml/Stool.0.03.subsample.0.03.filter.combined');       

xTr = d(1:100,1:end-1)';
yTr = d(1:100,end)';
xTe = d(101:end,1:end-1)';
yTe = d(101:end,end)';

% To speed up the demo, I run lmnn with k=1. 
% Sometimes k=3 is slightly better. 
%
                 
fprintf('Running single metric LMNN (with dimensionality reduction from 50d to 15d)  ...\n');
[Ldim,Det]=lmnn2(xTr,yTr,3,'outdim',15,'quiet',0,'maxiter',500,'validation',0.3,'checkup',0);
enerrdim=energyclassify(Ldim,xTr,yTr,xTe,yTe,3);
knnerrLdim=knnclassifytree(Ldim,xTr,yTr,xTe,yTe,3);
knnerrI=knnclassifytree(eye(size(xTr,1)),xTr,yTr,xTe,yTe,3);

fprintf('3-NN Euclidean training error: %2.2f\n',knnerrI(1)*100);
fprintf('3-NN Euclidean testing error: %2.2f\n',knnerrI(2)*100);
fprintf('15-dim usps digits data set (after dim-reduction):\n');
fprintf('3-NN Malhalanobis training error: %2.2f\n',knnerrLdim(1)*100);
fprintf('3-NN Malhalanobis testing error: %2.2f\n',knnerrLdim(2)*100);
fprintf('\nEnergy classification error: %2.2f\n',enerrdim*100);
fprintf('\nTraining time: %2.2fs\n\n\n',Det.time);


fprintf('Running single metric LMNN  ...\n');
[L,Det]=lmnn2(xTr,yTr,1,'quiet',1,'maxiter',500,'validation',0.3,'checkup',0);
enerr=energyclassify(L,xTr,yTr,xTe,yTe,3);
knnerrL=knnclassifytree(L,xTr,yTr,xTe,yTe,3);


clc;
fprintf('100-dim usps digits data set:\n');
fprintf('3-NN classification:\n')
fprintf('Training:\tEuclidean=%2.2f\t1-Metric(dim-red)=%2.2f\t1-Metric(square)=%2.2f\n',knnerrI(1)*100,knnerrLdim(1)*100,knnerrL(1)*100)
fprintf('Testing:\tEuclidean=%2.2f\t1-Metric(dim-red)=%2.2f\t1-Metric(square)=%2.2f\n',knnerrI(2)*100,knnerrLdim(2)*100,knnerrL(2)*100)
fprintf('\nEnergy classification:\n')
fprintf('Testing:\t1-Metric(dim-red)=%2.2f\t1-Metric(square)=%2.2f\n',enerrdim*100,enerr*100);
fprintf('\nTraining time: %2.2fs\n\n',Det.time);

% removed multiple-metrics LMNN

