function accSVM = acc_calculate(reducedMatrix,f_num,kfold)

% Wejście:
% 
% - reducedMatrix - wejściowa macierz danych
% - f_num - liczba cech wykorzystanych do klasyfikacji
% - kfold - liczba podzbiorów użytych do walidacji krzyżowej
% 
% Wyjście:
% 
% - accSVM - 
rng(10);
reducedMatrix = reducedMatrix(:,1:f_num+1);
c_k = cvpartition(reducedMatrix(:,1),'KFold',kfold);

reducedMatrix_class = reducedMatrix(:,1);
reducedMatrix_features = reducedMatrix(:,2:end);

% mdlSVM = fitcecoc(reducedMatrix(:,2:end),reducedMatrix(:,1),'Learners','svm','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('ShowPlots',false));
% c = mdlSVM.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
% sigma = mdlSVM.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;
% clear mdlSVM;
% t = templateSVM('BoxConstraint',c,'KernelScale',sigma);

accSVM = zeros(kfold,2);

for j = 1:kfold
    idxTest = test(c_k,j);
    idxTrain = training(c_k,j);
    
    fTrain = reducedMatrix_features(idxTrain,:);
    fTest = reducedMatrix_features(idxTest,:);
    
    cTrain = reducedMatrix_class(idxTrain,:);
    cTest = reducedMatrix_class(idxTest,:);
    
    % Budowa modelu
    mdlSVM_tmp = fitcecoc(fTrain,cTrain,'Learners','svm','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('ShowPlots',false));
    c = mdlSVM_tmp.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
    sigma = mdlSVM_tmp.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;
    clear mdlSVM_tmp;

    t = templateSVM('BoxConstraint',c,'KernelScale',sigma);
    mdl_best = fitcecoc(fTrain,cTrain,'Coding','onevsall','Learners',t);
    
    accSVM_tmp = 1 - loss(mdl_best,fTest,cTest);
    accSVM(j,1) = f_num;
    accSVM(j,2) = accSVM_tmp + accSVM(j,2);
    clear mdl_best;
end

end


