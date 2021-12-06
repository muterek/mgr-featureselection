function [reducedFeaturesMat, sortedFeatures_names] = LASSO_FSel(data,p,feat_names)

classVec = data(:,1); %class vector
features = data(:,2:end); % feature table

J = unique(classVec); % classes
if length(J)>2
    I = length(J);
else
    I = 1;
end

t = templateLinear('Learner','svm','Lambda','auto','Regularization','lasso','solver','sgd');
[Mdl,HyperparameterOptimizationResults] = fitcecoc(features,classVec,'Coding','onevsall','Learners',t);

Bin = zeros(size(features,2),1);
w_abs_sum = zeros(size(features,2),1);
for i=1:I
    tmpBin = Mdl.BinaryLearners{i}.Beta==0;
    Bin = Bin + tmpBin;
    
    w_abs_sum_tmp = abs(Mdl.BinaryLearners{i}.Beta);
    w_abs_sum = w_abs_sum + w_abs_sum_tmp;
end
w_abs_sum(Bin~=0) = 0;

[w_storted, index] = sort(w_abs_sum.^2,'descend');
sortedFeatures = features(:,index);
sortedFeatures_names = feat_names(index,1);

sortedFeatures = sortedFeatures(:,1:round((p/100)*(size(data,2)-1)));
sortedFeatures_names = sortedFeatures_names(1:round((p/100)*size(data(:,2:end),2)),1);

reducedFeaturesMat = [classVec, sortedFeatures];

end
