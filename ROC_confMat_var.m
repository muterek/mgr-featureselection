function [ROC,C,AUC_final,Tsvm_all, cTest_all, predictedLabels_all, Posterior_all] = ROC_confMat_var(k_fold,reducedMatrix,number_of_features)
rng(10);
c_k = cvpartition(reducedMatrix(:,1),'KFold',k_fold);

mdlSVM = fitcecoc(reducedMatrix(:,2:end),reducedMatrix(:,1),'Learners','svm','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('ShowPlots',false));
c = mdlSVM.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
sigma = mdlSVM.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;
clear mdlSVM;
t = templateSVM('BoxConstraint',c,'KernelScale',sigma);

bestMatrix = reducedMatrix(:,1:number_of_features+1);

bestMatrix_class = bestMatrix(:,1);
bestMatrix_features = bestMatrix(:,2:end);

J = unique(bestMatrix_class); % classes
C = zeros(length(J),length(J));
% Xsvm_all = [];
% Ysvm_all = [];

cTest_all = [];
predictedLabels_all = [];
Posterior_all = [];

for i  = 1:k_fold
    idxTest = test(c_k,i);
    idxTrain = training(c_k,i);
    
    fTrain = bestMatrix_features(idxTrain,:);
    fTest = bestMatrix_features(idxTest,:);
    
    cTrain = bestMatrix_class(idxTrain,:);
    cTest = bestMatrix_class(idxTest,:);
    
    % Budowa modelu
    mdl_best = fitcecoc(fTrain,cTrain,'Coding','onevsall','Learners',t,'FitPosterior',true);
    
    % Wyznaczenie prawdopodobieństw przynależności do klas
    [predictedLabels,NegLoss,PBScore,Posterior] = predict(mdl_best, fTest);
    
    % Macierz pomyłek
    C_tmp = confusionmat(cTest,predictedLabels);
    for j = 1:length(J)
        if size(C_tmp,1)<length(J) && (sum(unique(cTest)==j) == 0) && (sum(unique(predictedLabels)==j) == 0)
            C_tmp = [C_tmp(:,1:j-1), zeros(length(C_tmp),1), C_tmp(:,j:end)];
            C_tmp = [C_tmp(1:j-1,:); zeros(1,size(C_tmp,2)); C_tmp(j:end,:)];
        end
    end
    C = C + C_tmp;
    
    % ROC
    Xsvm = [];
    Ysvm = [];
    Tsvm = [];
    AUCsvm = [];
    
%     for j=1:length(J)
%         [Xsvm(:,j),Ysvm(:,j),Tsvm(:,j),AUCsvm(j)] = perfcurve(cTest,Posterior(:,j),num2str(j));
%     end
    disp(num2str(i));
%     Xsvm_mean = sum(Xsvm,2)./length(J);
%     Xsvm_all = [Xsvm_all; Xsvm];
%     Xsvm_all = [Xsvm_all; Xsvm_mean];
%     Ysvm_mean = sum(Ysvm,2)./length(J);
%     Ysvm_all = [Ysvm_all; Ysvm_mean];
%     Ysvm_all = [Ysvm_all; Ysvm];
%     AUCsvm_all{i} = AUCsvm;
    Tsvm_all{i} = Tsvm;
%     
    cTest_all = [cTest_all; cTest];
    Posterior_all = [Posterior_all; Posterior];

    predictedLabels_all = [predictedLabels_all; predictedLabels];
    
end

for j=1:length(J)
    [Xsvm_final(:,j),Ysvm_final(:,j),Tsvm_final(:,j),AUC_final(j)] = perfcurve(cTest_all,Posterior_all(:,j),num2str(j));
end

Xsvm_final = sum(Xsvm_final,2)./length(J);
Ysvm_final = sum(Ysvm_final,2)./length(J);
AUC_final = trapz(Xsvm_final,Ysvm_final);
ROC = [Xsvm_final, Ysvm_final];
ROC = sort(ROC);
end

