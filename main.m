%% Czyszczenie pamięci
clear; 
close all;
rng(10);
%% Załadowanie danych
todays_date = datestr(datetime('today'));
data_set = 'Leukemia.txt';

[datat, data, classVec, features, data_name, feat_names] = read_file(data_set);
class_names = datat{1,1};
class_names = strsplit(string(class_names),'#');
class_names = class_names(~cellfun('isempty',class_names));
fpath = 'D:\Studia_Elka\MAGISTERKA\SVM_program\wyniki_wykresy3\';

%% PCA - Przedstawienie danych (pierwsza i druga składowa główna)

w = 1./var(features); %When the variables are in different units or the difference in the variance of different columns is substantial, scaling of the data or use of weights is often preferable
tic
[wcoeff,score,latent,tsquared,explained,mu] = pca(features,'VariableWeights',w);
time_PCA = toc;

% Wykres 1 i 2 składowej głównej
classes = unique(classVec); % klasy
Fig_PCA = figure('Color','white');
for i = 1:length(classes)
    plot(score(classVec==i,1),score(classVec==i,2),'o','DisplayName', [class_names{i}])
    hold on
end
legend show
title([strrep(data_name,'_','-') ': wykres dwóch pierwszych składowych głównych']);
xlabel('Pierwsza składowa główna')
ylabel('Druga składowa główna')
hold off

filename = append(data_name,'_',todays_date,'_wykres_2_skladowych_glownych');
exportgraphics(Fig_PCA, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(Fig_PCA,fullfile(fpath, append(filename,'.fig')));

% Wykres Pareto
Fig_Pareto = figure('Color','white');
pareto(explained(1:5,:))
title([strrep(data_name,'_','-') ': wykres Pareto']);
xlabel('Składowe główne') 

filename = append(data_name,'_',todays_date,'_wykres_Pareto');
exportgraphics(Fig_Pareto, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(Fig_Pareto,fullfile(fpath, append(filename,'.fig')));

%% SELEKCJA CECH

% POJEDYNCZE (metryka Fishera, LASSO, SVM-RFE)
p = 100; % procent wybranych cech

% metryka Fishera
disp('Wyznaczanie macierzy cech uszeregowanych przy pomocy metryki Fishera...');
tic
[reducedFeaturesMat_Fscore, reducedFeaturesMat_names_Fscore] = Fscore(data,p, feat_names); % 1 vs. rest
time_Fscore = toc;
disp('Wyznaczono nową macierz cech przy pomocy metryki Fishera.');

% SVM-RFE
disp('Wyznaczanie macierzy cech uszeregowanych przy pomocy SVM-RFE...');
p = 100; % procent wybranych cech
tic
[reducedFeaturesMat_SVMRFE, reducedFeaturesMat_names_SVMRFE]  = linear_SVM_RFE(data,p,table2cell(feat_names));
time_SVMRFE = toc;
disp('Wyznaczono nową macierz cech przy pomocy SVM-RFE.');

% LASSO

disp('Wyznaczanie macierzy cech uszeregowanych przy pomocy LASSO...');
p = 100; % procent wybranych cech
tic
[reducedFeaturesMat_LASSO,reducedFeaturesMat_names_LASSO] = LASSO_FSel(data,p,table2cell(feat_names));
time_LASSO = toc;
disp('Wyznaczono nową macierz cech przy pomocy LASSO.');

% PODWÓJNE (Fscore + SVM-RFE, LASSO + SVM-RFE)

% Fscore i SVM-RFE
disp('Wyznaczanie macierzy cech uszeregowanych przy pomocy metryki Fishera i SVM-RFE...');
p = 100; % procent wybranych cech
tic
[reducedFeaturesMat_Fscore_SVMRFE, reducedFeaturesMat_names_Fscore_SVMRFE] = Fscore(data,p, feat_names); % 1 vs. rest
if size(reducedFeaturesMat_Fscore_SVMRFE,2)-1>=2500
    reducedFeaturesMat_Fscore_SVMRFE = reducedFeaturesMat_Fscore_SVMRFE(:,1:2501);
    reducedFeaturesMat_names_Fscore_SVMRFE = reducedFeaturesMat_names_Fscore_SVMRFE(1:2500,1);
else
    reducedFeaturesMat_Fscore_SVMRFE = reducedFeaturesMat_Fscore_SVMRFE(:,1:end);
    reducedFeaturesMat_names_Fscore_SVMRFE = reducedFeaturesMat_names_Fscore_SVMRFE(1:end,1);
end
[reducedFeaturesMat_Fscore_SVMRFE, reducedFeaturesMat_names_Fscore_SVMRFE]  = linear_SVM_RFE(reducedFeaturesMat_Fscore_SVMRFE,p,reducedFeaturesMat_names_Fscore_SVMRFE);
time_Fscore_SVMRFE = toc;
disp('Wyznaczono nową macierz cech przy pomocy metryki Fishera i SVM-RFE.');

% LASSO i SVM-RFE
disp('Wyznaczanie macierzy cech uszeregowanych przy pomocy LASSO i SVM-RFE...');
p = 100; % procent wybranych cech
tic
[reducedFeaturesMat_LASSO_SVMRFE, reducedFeaturesMat_names_LASSO_SVMRFE] = LASSO_FSel(data,p, table2cell(feat_names)); % 1 vs. rest
if size(reducedFeaturesMat_LASSO_SVMRFE,2)-1>=2500
    reducedFeaturesMat_LASSO_SVMRFE = reducedFeaturesMat_LASSO_SVMRFE(:,1:2501);
    reducedFeaturesMat_names_LASSO_SVMRFE = reducedFeaturesMat_names_LASSO_SVMRFE(1:2500,1);
else
    reducedFeaturesMat_LASSO_SVMRFE = reducedFeaturesMat_LASSO_SVMRFE(:,1:end);
    reducedFeaturesMat_names_LASSO_SVMRFE = reducedFeaturesMat_names_LASSO_SVMRFE(1:end,1);
end

[reducedFeaturesMat_LASSO_SVMRFE, reducedFeaturesMat_names_LASSO_SVMRFE]  = linear_SVM_RFE(reducedFeaturesMat_LASSO_SVMRFE,p,reducedFeaturesMat_names_LASSO_SVMRFE);
time_LASSO_SVMRFE = toc;
disp('Wyznaczono nową macierz cech przy pomocy metryki Fishera i SVM-RFE.');

%% porównanie metod selekcji cech ROC/dokladnosc dla 50, 100,... cech
% Wykres boxplot
rng(10);
kfold = 5;

% Pojedyncze
f = [10 50 100 200 300 500 1000 2500 size(features,2)];
accSVM_F = [];
accSVM_S = [];
accSVM_L = [];
accSVM_bef = [];

for i = f
    accSVM_F_tmp = acc_calculate(reducedFeaturesMat_Fscore,i,kfold);
    accSVM_F = [accSVM_F; accSVM_F_tmp];
    accSVM_S_tmp = acc_calculate(reducedFeaturesMat_SVMRFE,i,kfold);
    accSVM_S = [accSVM_S; accSVM_S_tmp];
    accSVM_L_tmp = acc_calculate(reducedFeaturesMat_LASSO,i,kfold);
    accSVM_L = [accSVM_L; accSVM_L_tmp];
    accSVM_bef_tmp = acc_calculate(data,i,kfold);
    accSVM_bef = [accSVM_bef; accSVM_bef_tmp];
end

% Połączone
f1 = [10 50 100 200 300 500 1000 2000 2500];
accSVM_FS = [];
accSVM_LS = [];
accSVM_S500 = [];

for i = f1
    accSVM_FS_tmp = acc_calculate(reducedFeaturesMat_Fscore_SVMRFE,i,kfold);
    accSVM_FS = [accSVM_FS; accSVM_FS_tmp];
    accSVM_LS_tmp = acc_calculate(reducedFeaturesMat_LASSO_SVMRFE,i,kfold);
    accSVM_LS = [accSVM_LS; accSVM_LS_tmp];
    accSVM_S500_tmp = acc_calculate(reducedFeaturesMat_SVMRFE,i,kfold);
    accSVM_S500 = [accSVM_S500; accSVM_S500_tmp];
end

%% Porównanie metod selekcji cech ROC/dokladnosc dla 50, 100,... cech
% Wykres punktowy

% Pojedyncze
acc_err_single = figure('Color','white');
x = f;
tmp = accSVM_F(:,2).*(accSVM_F(:,1)==f);
tmp(tmp==0)=NaN;
yF = mean(tmp,'omitnan'); 
tmp = accSVM_S(:,2).*(accSVM_S(:,1)==f);
tmp(tmp==0)=NaN;
yS = mean(tmp,'omitnan'); 
tmp = accSVM_L(:,2).*(accSVM_L(:,1)==f);
tmp(tmp==0)=NaN;
yL = mean(tmp,'omitnan'); 
tmp = accSVM_bef(:,2).*(accSVM_bef(:,1)==f);
tmp(tmp==0)=NaN;
ybef = mean(tmp,'omitnan');
plot(x, yF,'--')
hold on
plot(x, yS,'--')
hold on
plot(x, yL,'--')
hold on
plot(x, ybef,'--')
legend('Fscore', 'SVM-RFE', 'LASSO','losowy wybór','Location','southeast');
ylabel('Dokładność klasyfikacji');
xlabel('Liczba użytych cech');
xlim([5 100000])
set(gca,'XScale','log')
set(gca,'YScale','linear')

filename = append(data_name,'_',todays_date,'_wykres_acc_err_pojedyncze');
exportgraphics(acc_err_single, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(acc_err_single,fullfile(fpath, append(filename,'.fig')));

% Połączone
acc_err_fused = figure('Color','white');
x = f1;
tmp = accSVM_FS(:,2).*(accSVM_FS(:,1)==f1);
tmp(tmp==0)=NaN;
yFS = mean(tmp,'omitnan'); 
tmp = accSVM_LS(:,2).*(accSVM_LS(:,1)==f1);
tmp(tmp==0)=NaN;
yLS = mean(tmp,'omitnan'); 
tmp = accSVM_S500(:,2).*(accSVM_S500(:,1)==f1);
tmp(tmp==0)=NaN;
yS500 = mean(tmp,'omitnan');
plot(x, yS500,'--')
hold on
plot(x, yFS,'--')
hold on
plot(x, yLS,'--')
legend('SVM-RFE','Fscore + SVM-RFE', 'LASSO + SVM-RFE','Location','southeast');
ylabel('Dokładność klasyfikacji');
xlabel('Liczba użytych cech');
xlim([5 10000]);
set(gca,'XScale','log')
set(gca,'YScale','linear')

filename = append(data_name,'_',todays_date,'_wykres_acc_err_poloczone');
exportgraphics(acc_err_fused, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(acc_err_fused,fullfile(fpath, append(filename,'.fig')));

%% Wybór ilości cech do ROC

n_min_Fscore = search_min(reducedFeaturesMat_Fscore);
n_min_SVMRFE = search_min(reducedFeaturesMat_SVMRFE);
n_min_LASSO = search_min(reducedFeaturesMat_LASSO);
n_min_Fscore_SVMRFE = search_min(reducedFeaturesMat_Fscore_SVMRFE);
n_min_LASSO_SVMRFE = search_min(reducedFeaturesMat_LASSO_SVMRFE);

%% ---------Porównanie metod selekcji cech (wykres ROC dla min błędu)---------
k_fold = 5; % podział danych na k_fold podzbiorów

% ---------- Wyniki klasyfikacji przed selekcją cech ----------
[ROC_bef,C,AUCsvm_bef,Tsvm_bef, cTest_bef, predictedLabels_bef, Posterior_bef] = ROC_confMat_var(5,data,length(features));
metricsTable_bef = confMat_metrics(cTest_bef,predictedLabels_bef);

% POJEDYNCZE
% ---------------- F-score ----------------
[ROC_Fscore,C_Fscore,AUCsvm_all_Fscore,Tsvm_all_Fscore,cTest_Fscore, predictedLabels_Fscore, Posterior_Fscore] = ROC_confMat_var(k_fold,reducedFeaturesMat_Fscore,n_min_Fscore);

% ---------------- SVM-RFE ----------------
[ROC_SVMRFE,C_SVMRFE,AUCsvm_all_SVMRFE,Tsvm_all_SVMRFE, cTest_SVMRFE, predictedLabels_SVMRFE, Posterior_SVMRFE] = ROC_confMat_var(k_fold,reducedFeaturesMat_SVMRFE,n_min_SVMRFE);

% ----------------- LASSO -----------------
[ROC_LASSO,C_LASSO,AUCsvm_all_LASSO,Tsvm_all_LASSO, cTest_LASSO, predictedLabels_LASSO, Posterior_LASSO] = ROC_confMat_var(k_fold,reducedFeaturesMat_LASSO,n_min_LASSO);

Fig_ROC1 = figure('Color','white');
plot(ROC_Fscore(:,1),ROC_Fscore(:,2))
hold on
plot(ROC_SVMRFE(:,1),ROC_SVMRFE(:,2))
hold on
plot(ROC_LASSO(:,1),ROC_LASSO(:,2))
hold on
plot(ROC_bef(:,1),ROC_bef(:,2))
legend('Fscore', 'SVM-RFE','LASSO', 'przed selekcją')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve macro average')
hold off

filename = append(data_name,'_',todays_date,'_wykres_ROC1');
exportgraphics(Fig_ROC1, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(Fig_ROC1,fullfile(fpath, append(filename,'.fig')));

% PODWÓJNE
% ----------------- Fscore + SVMRFE -----------------
[ROC_Fscore_SVMRFE,C_Fscore_SVMRFE,AUCsvm_all_Fscore_SVMRFE,Tsvm_all_Fscore_SVMRFE, cTest_Fscore_SVMRFE, predictedLabels_Fscore_SVMRFE, Posterior_Fscore_SVMRFE] = ROC_confMat_var(k_fold,reducedFeaturesMat_Fscore_SVMRFE,n_min_Fscore_SVMRFE);

% ----------------- LASSO + SVMRFE -----------------
[ROC_LASSO_SVMRFE,C_LASSO_SVMRFE,AUCsvm_all_LASSO_SVMRFE,Tsvm_all_LASSO_SVMRFE, cTest_LASSO_SVMRFE, predictedLabels_LASSO_SVMRFE, Posterior_LASSO_SVMRFE] = ROC_confMat_var(k_fold,reducedFeaturesMat_LASSO_SVMRFE,n_min_LASSO_SVMRFE);

Fig_ROC2 = figure('Color','white');
plot(ROC_Fscore_SVMRFE(:,1),ROC_Fscore_SVMRFE(:,2))
hold on
plot(ROC_LASSO_SVMRFE(:,1),ROC_LASSO_SVMRFE(:,2))
hold on
plot(ROC_SVMRFE(:,1),ROC_SVMRFE(:,2))
legend('Fscore + SVMRFE','LASSO + SVMRFE','SVM-RFE')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve macro average')
hold off

filename = append(data_name,'_',todays_date,'_wykres_ROC2');
exportgraphics(Fig_ROC2, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(Fig_ROC2,fullfile(fpath, append(filename,'.fig')));

% ROC Curve
Fig_ROC = figure('Color','white');
plot(ROC_Fscore(:,1),ROC_Fscore(:,2))
hold on
plot(ROC_SVMRFE(:,1),ROC_SVMRFE(:,2))
hold on
plot(ROC_LASSO(:,1),ROC_LASSO(:,2))
hold on
plot(ROC_Fscore_SVMRFE(:,1),ROC_Fscore_SVMRFE(:,2))
hold on
plot(ROC_LASSO_SVMRFE(:,1),ROC_LASSO_SVMRFE(:,2))
hold on
plot(ROC_bef(:,1),ROC_bef(:,2))
legend('Fscore', 'SVM-RFE','LASSO','Fscore + SVMRFE','LASSO + SVMRFE','przed selekcją')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve macro average')
hold off

filename = append(data_name,'_',todays_date,'_wykres_ROC');
exportgraphics(Fig_ROC, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(Fig_ROC,fullfile(fpath, append(filename,'.fig')));

%% Czy cechy między metodami się powatrzają?
n = 200;

frep_F_S = sum(ismember(string(reducedFeaturesMat_names_Fscore(1:n)),string(reducedFeaturesMat_names_SVMRFE(1:n))));
frep_F_L = sum(ismember(string(reducedFeaturesMat_names_Fscore(1:n)),string(reducedFeaturesMat_names_LASSO(1:n))));
frep_L_S = sum(ismember(string(reducedFeaturesMat_names_LASSO(1:n)),string(reducedFeaturesMat_names_SVMRFE(1:n))));

frep_F_FS = sum(ismember(string(reducedFeaturesMat_names_Fscore(1:n)),string(reducedFeaturesMat_names_Fscore_SVMRFE(1:n))));
frep_F_LS = sum(ismember(string(reducedFeaturesMat_names_Fscore(1:n)),string(reducedFeaturesMat_names_LASSO_SVMRFE(1:n))));

frep_S_FS = sum(ismember(string(reducedFeaturesMat_names_SVMRFE(1:n)),string(reducedFeaturesMat_names_Fscore_SVMRFE(1:n))));
frep_S_LS = sum(ismember(string(reducedFeaturesMat_names_SVMRFE(1:n)),string(reducedFeaturesMat_names_LASSO_SVMRFE(1:n))));

frep_L_FS = sum(ismember(string(reducedFeaturesMat_names_LASSO(1:n)),string(reducedFeaturesMat_names_Fscore_SVMRFE(1:n))));
frep_L_LS = sum(ismember(string(reducedFeaturesMat_names_LASSO(1:n)),string(reducedFeaturesMat_names_LASSO_SVMRFE(1:n))));

frep_FS_LS = sum(ismember(string(reducedFeaturesMat_names_Fscore_SVMRFE(1:n)),string(reducedFeaturesMat_names_LASSO_SVMRFE(1:n))));

lab = {'Fscore', 'SVM-RFE', 'LASSO', 'Fscore+SVM-RFE', 'LASSO+SVM-RFE'};
mat = [n, frep_F_S, frep_F_L, frep_F_FS, frep_F_LS;
    frep_F_S, n, frep_L_S, frep_S_FS, frep_S_LS;
    frep_F_L, frep_L_S, n, frep_L_FS, frep_L_LS;
    frep_F_FS, frep_S_FS, frep_L_FS, n, frep_FS_LS;
    frep_F_LS, frep_S_LS, frep_L_LS, frep_FS_LS, n];    
h = figure('Color','white');
heatmap(lab,lab,mat);

filename = append(data_name,'_',todays_date,'_heatmap');
exportgraphics(h, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(h,fullfile(fpath, append(filename,'.fig')));

%% --------------------- PCA PRZED i PO SELEKCJI CECH ---------------------  

% wyniki klasyfikacji liniowego modelu SVM przy wykorzystaniu skł. głównych

% przed selekcją cech
PCA_matrix = [classVec score];
stepPCA = 2;
n_max = size(PCA_matrix,2);

% Fscore
wF = 1./var(reducedFeaturesMat_Fscore(:,2:n_min_Fscore)); %When the variables are in different units or the difference in the variance of different columns is substantial, scaling of the data or use of weights is often preferable
[wcoeffF,scoreF,latentF,tsquaredF,explainedF,muF] = pca(reducedFeaturesMat_Fscore(:,2:n_min_Fscore),'VariableWeights',wF);
PCA_matrix_F = [classVec scoreF];

% SVMRFE
wS = 1./var(reducedFeaturesMat_SVMRFE(:,2:n_min_SVMRFE)); %When the variables are in different units or the difference in the variance of different columns is substantial, scaling of the data or use of weights is often preferable
[wcoeffS,scoreS,latentS,tsquaredS,explainedS,muS] = pca(reducedFeaturesMat_SVMRFE(:,2:n_min_SVMRFE),'VariableWeights',wS);
PCA_matrix_S = [classVec scoreS];

% LASSO
wL = 1./var(reducedFeaturesMat_LASSO(:,2:n_min_LASSO)); %When the variables are in different units or the difference in the variance of different columns is substantial, scaling of the data or use of weights is often preferable
[wcoeffL,scoreL,latentL,tsquaredL,explainedL,muL] = pca(reducedFeaturesMat_LASSO(:,2:n_min_LASSO),'VariableWeights',wL);
PCA_matrix_L = [classVec scoreL];

% Fscore + SVMRFE
wFS = 1./var(reducedFeaturesMat_Fscore_SVMRFE(:,2:n_min_Fscore_SVMRFE)); %When the variables are in different units or the difference in the variance of different columns is substantial, scaling of the data or use of weights is often preferable
[wcoeffFS,scoreFS,latentFS,tsquaredFS,explainedFS,muFS] = pca(reducedFeaturesMat_Fscore_SVMRFE(:,2:n_min_Fscore_SVMRFE),'VariableWeights',wFS);
PCA_matrix_FS = [classVec scoreFS];

% LASSO + SVMRFE
wLS = 1./var(reducedFeaturesMat_LASSO_SVMRFE(:,2:n_min_LASSO_SVMRFE)); %When the variables are in different units or the difference in the variance of different columns is substantial, scaling of the data or use of weights is often preferable
[wcoeffLS,scoreLS,latentLS,tsquaredLS,explainedLS,muLS] = pca(reducedFeaturesMat_LASSO_SVMRFE(:,2:n_min_LASSO_SVMRFE),'VariableWeights',wLS);
PCA_matrix_LS = [classVec scoreLS];

classes = unique(classVec); % klasy
figure('Color','white');
for i = 1:length(classes)
    plot(scoreLS(classVec==i,1),scoreLS(classVec==i,2),'o','DisplayName', ['klasa ' num2str(i)])
    hold on
end
legend show
title([strrep(data_name,'_','-') ': wykres dwóch pierwszych składowych głównych']);
xlabel('Pierwsza składowa główna')
ylabel('Druga składowa główna')
hold off

%% Wykres ROC po PCA dla każdej z metod selekcji cech

k_fold = 5;
[ROC_PCA,C_PCA,AUCsvm_all_PCA,Tsvm_all_PCA, cTest_PCA, predictedLabels_PCA, Posterior_PCA] = ROC_confMat_var(k_fold,PCA_matrix,size(PCA_matrix,2)-1);
[ROC_PCA_F,C_PCA_F,AUCsvm_all_PCA_F,Tsvm_all_PCA_F, cTest_PCA_F, predictedLabels_PCA_F, Posterior_PCA_F] = ROC_confMat_var(k_fold,PCA_matrix_F,size(PCA_matrix_F,2)-1);
[ROC_PCA_S,C_PCA_S,AUCsvm_all_PCA_S,Tsvm_all_PCA_S, cTest_PCA_S, predictedLabels_PCA_S, Posterior_PCA_S] = ROC_confMat_var(k_fold,PCA_matrix_S,size(PCA_matrix_S,2)-1);
[ROC_PCA_L,C_PCA_L,AUCsvm_all_PCA_L,Tsvm_all_PCA_L, cTest_PCA_L, predictedLabels_PCA_L, Posterior_PCA_L] = ROC_confMat_var(k_fold,PCA_matrix_L,size(PCA_matrix_L,2)-1);
[ROC_PCA_FS,C_PCA_FS,AUCsvm_all_PCA_FS,Tsvm_all_PCA_FS, cTest_PCA_FS, predictedLabels_PCA_FS, Posterior_PCA_FS] = ROC_confMat_var(k_fold,PCA_matrix_FS,size(PCA_matrix_FS,2)-1);
[ROC_PCA_LS,C_PCA_LS,AUCsvm_all_PCA_LS,Tsvm_all_PCA_LS, cTest_PCA_LS, predictedLabels_PCA_LS, Posterior_PCA_LS] = ROC_confMat_var(k_fold,PCA_matrix_LS,size(PCA_matrix_LS,2)-1);

Fig_ROC_PCA = figure('Color','white');
plot(ROC_PCA_F(:,1),ROC_PCA_F(:,2))
hold on
plot(ROC_PCA_S(:,1),ROC_PCA_S(:,2))
hold on
plot(ROC_PCA_L(:,1),ROC_PCA_L(:,2))
hold on
plot(ROC_PCA_FS(:,1),ROC_PCA_FS(:,2))
hold on
plot(ROC_PCA_LS(:,1),ROC_PCA_LS(:,2))
hold on
plot(ROC_PCA(:,1),ROC_PCA(:,2))
legend('Fscore', 'SVM-RFE','LASSO','Fscore + SVMRFE','LASSO + SVMRFE','przed selekcją')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve macro average')
hold off

filename = append(data_name,'_',todays_date,'_wykres_ROC_PCA');
exportgraphics(Fig_ROC_PCA, [fpath, append(filename,'.jpeg')],'Resolution',600);
saveas(Fig_ROC_PCA,fullfile(fpath, append(filename,'.fig')));

%% ---------- metryki oceny modelu ----------------

metricsTable_Fscore = confMat_metrics(cTest_Fscore,predictedLabels_Fscore);
metricsTable_SVMRFE = confMat_metrics(cTest_SVMRFE,predictedLabels_SVMRFE);
metricsTable_LASSO = confMat_metrics(cTest_LASSO,predictedLabels_LASSO);

metricsTable_Fscore_SVMRFE = confMat_metrics(cTest_Fscore_SVMRFE,predictedLabels_Fscore_SVMRFE);
metricsTable_LASSO_SVMRFE = confMat_metrics(cTest_LASSO_SVMRFE,predictedLabels_LASSO_SVMRFE);

metricsTable_PCA = confMat_metrics(cTest_PCA,predictedLabels_PCA);
metricsTable_PCA_F = confMat_metrics(cTest_PCA_F,predictedLabels_PCA_F);
metricsTable_PCA_S = confMat_metrics(cTest_PCA_S,predictedLabels_PCA_S);
metricsTable_PCA_L = confMat_metrics(cTest_PCA_L,predictedLabels_PCA_L);
metricsTable_PCA_FS = confMat_metrics(cTest_PCA_FS,predictedLabels_PCA_FS);
metricsTable_PCA_LS = confMat_metrics(cTest_PCA_LS,predictedLabels_PCA_LS);

timeTable = [time_Fscore,time_SVMRFE,time_LASSO,time_Fscore_SVMRFE,time_LASSO_SVMRFE,time_PCA];
timeTable=array2table(timeTable,'RowNames',{'time'},'VariableNames', {'Fscore','SVMRFE','LASSO','Fscore + SVMRFE','LASSO + SVMRFE','PCA'});

meanAUCTable = [mean(AUCsvm_bef),...
    mean(AUCsvm_all_Fscore),...
    mean(AUCsvm_all_SVMRFE),...
    mean(AUCsvm_all_LASSO),...
    mean(AUCsvm_all_Fscore_SVMRFE),...
    mean(AUCsvm_all_LASSO_SVMRFE),...
    mean(AUCsvm_all_PCA),...
    mean(AUCsvm_all_PCA_F),...
    mean(AUCsvm_all_PCA_S),...
    mean(AUCsvm_all_PCA_L),...
    mean(AUCsvm_all_PCA_FS),...
    mean(AUCsvm_all_PCA_LS)];
meanAUCTable=array2table(meanAUCTable,'RowNames',{'time'},'VariableNames', ...
    {'Przed selekcją',...
    'Fscore','SVMRFE','LASSO',...
    'Fscore + SVMRFE','LASSO + SVMRFE',...
    'PCA','PCA_F','PCA_S','PCA_L','PCA_FS','PCA_LS'});
filename = append(data_name,'_',todays_date,'_metryki_oceny_good');
writetable(metricsTable_bef,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','Before')
writetable(metricsTable_Fscore,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','Fscore')
writetable(metricsTable_SVMRFE,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','SVM-RFE')
writetable(metricsTable_LASSO,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','LASSO')
writetable(metricsTable_Fscore_SVMRFE,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','Fscore_SVMRFE')
writetable(metricsTable_LASSO_SVMRFE,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','LASSO_SVMRFE')
writetable(metricsTable_PCA,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','PCA')
writetable(metricsTable_PCA_F,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','PCA_Fscore')
writetable(metricsTable_PCA_S,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','PCA_SVMRFE')
writetable(metricsTable_PCA_L,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','PCA_LASSO')
writetable(metricsTable_PCA_FS,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','PCA_FS')
writetable(metricsTable_PCA_LS,[fpath, append(filename,'.xlsx')],'WriteRowNames',1,'Sheet','PCA_LS')
writetable(timeTable,[fpath, append(filename,'.xlsx')],'Sheet','Time')
writetable(meanAUCTable,[fpath, append(filename,'.xlsx')],'Sheet','AUC')

filename = append(data_name,'_',todays_date,'_zmienne_good2');
save([fpath, append(filename,'.mat')])