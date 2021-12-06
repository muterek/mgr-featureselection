
function [lossSVM, X] = FSel_results(reducedFeaturesMat,kernel,k_fold,step,n_min,n_max)

c_k = cvpartition(reducedFeaturesMat(:,1),'KFold',k_fold);

if size(reducedFeaturesMat,2)<=n_max
    x=size(reducedFeaturesMat,2)-1;
else
    x=n_max;
end

X = n_min:step:x;

lossSVM = zeros(length(X),1);

j = 1;

for f = n_min:step:x
    disp([num2str(j) '/' num2str(length(X))]);
    
    reducedFeaturesMat_tmp = reducedFeaturesMat(:,1:f+1);
    classVec_tmp = reducedFeaturesMat_tmp(:,1); %class vector
    features_tmp = reducedFeaturesMat_tmp(:,2:end); % feature table
    
    for i  = 1:k_fold
        idxTest = test(c_k,i);
        idxTrain = training(c_k,i);
        
        fTrain = features_tmp(idxTrain,:);
        fTest = features_tmp(idxTest,:);
        cTrain = classVec_tmp(idxTrain,:);
        cTest = classVec_tmp(idxTest,:);
        
        mdlSVM = fitcecoc(fTrain,cTrain,'Learners','svm','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('ShowPlots',false));
        c = mdlSVM.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
        sigma = mdlSVM.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;
        clear mdlSVM;
        t = templateSVM('BoxConstraint',c,'KernelFunction',kernel,'KernelScale',sigma);
        
        mdlSVM = fitcecoc(fTrain,cTrain,'Learners',t);
        lossSVM_tmp = loss(mdlSVM,fTest,cTest);
        lossSVM(j) = lossSVM_tmp + lossSVM(j);
        clear mdlSVM;
    end
    
    j = j+1;
end

lossSVM = lossSVM./k_fold;
end
