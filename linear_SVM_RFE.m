function [reducedFeaturesMat, reducedFeaturesMat_names] = linear_SVM_RFE(data,p,feat_names)
data = data(:,1:ceil((p/100)*(size(data,2))));
classVec = data(:,1); %class vector
features = data(:,2:end); % feature table
feat_names = feat_names(1:ceil((p/100)*size(data(:,2:end),2)),:);

t = templateSVM('KernelFunction','linear');
mdlSVM = fitcecoc(features,classVec,'Learners',t,'OptimizeHyperparameters',{'KernelScale','BoxConstraint'},'HyperparameterOptimizationOptions',struct('ShowPlots',false));
c = mdlSVM.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint;
sigma = mdlSVM.HyperparameterOptimizationResults.XAtMinObjective.KernelScale;
% c=10;sigma=10;
clear t;
clear mdlSVM;

J = unique(classVec); % classes

% Linear SVM RFE
rank = [];
reducedFeaturesMat_names = [];

while size(features,2)>=2
    
    fTrain = features;
    %     fTest = features(idxTest,:);
    cTrain = classVec;
    %     cTest = classVec(idxTest,:);
    
    t = templateSVM('BoxConstraint',c,'KernelScale',sigma,'KernelFunction','linear','Standardize',false,'SaveSupportVectors',true);
    mdlSVM = fitcecoc(fTrain,cTrain,'Learners',t,'coding','onevsone');
    %     predictedGroups = predict(mdlSVM,fTest);
    %     [confSVM, grpSVM] = confusionmat(cTest,predictedGroups);
    
    number_of_coeff = J(end)*(J(end) - 1)/2; % OVO (One vs. One method)
    
    alpha = cell(number_of_coeff,1); %gives the multiplication between the class label (+1 or -1) and the alpha of each support vector
    SV = cell(number_of_coeff,1);
    SVLabels = cell(number_of_coeff,1);
    
    for i = 1 : number_of_coeff
        alpha{i} = mdlSVM.BinaryLearners{i}.Alpha;
        SV{i} = mdlSVM.BinaryLearners{i}.SupportVectors;
        SVLabels{i} = mdlSVM.BinaryLearners{i}.SupportVectorLabels;
    end
    w = zeros(1,size(features,2));
    for i = 1 : number_of_coeff
        w = w + sum(SV{i}.*(alpha{i}.*SVLabels{i}));
    end
    w = w.^2;
    [w_storted, index] = sort(w,'descend');
    reducedFeaturesMat_names = [{feat_names{index(end),1}}; reducedFeaturesMat_names];
    rank = [features(:,index(end)) rank];
    features(:,index(end)) = [];
    feat_names(index(end),:) = [];
    disp(['Zostało do posortowania ' num2str(size(features,2)) ' cech.']);
end
rank = [features rank];
reducedFeaturesMat_names = [{feat_names}; reducedFeaturesMat_names];
% reducedFeaturesMat_names = table2cell( reducedFeaturesMat_names );
% reducedFeaturesMat = rank(:,1:round((p/100)*(size(data,2)-1)));
% reducedFeaturesMat_names = reducedFeaturesMat_names(1:round((p/100)*length(size(data(:,2:end),2))));
reducedFeaturesMat = [classVec, rank];
end

