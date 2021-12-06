function [reducedFeaturesMat,reducedFeaturesMat_names] = Fscore(data, p, feat_names)

% F-score
% Input:
% - tabela danych (1 kolumna - klasa, reszta kolumn cechy)
% - procent cech o najlepszym f-score
% Output:
% - tabela danych ze zredukowan¹ iloœci¹ cech

classVec = data(:,1);
features = data(:,2:end); % Tabela cech

k = unique(data(:,1)); % Iloœæ klas
k_binary = dummyvar(data(:,1));

Fsc = zeros(1,length(features));

for j1 = 1:length(k)
    
    f1 = features(k_binary(:,j1)==1, :); %macierz cech aktualnej klasy
    fRest = features(k_binary(:,j1)==0, :); %macierz cech pozosta³ych klas
    
    mf1 = mean(f1); %œrednia aktualnej klasy
    mRest = mean(fRest); %œrednia pozosta³ych klas
    vf1 = var(f1); % var. aktualnej klasy
    vRest = var(fRest); % var. pozosta³ych klas
    
    Ftmp = (mf1-mRest).^2./(vf1+vRest); % Fscore (1 vs. rest) dla aktualnej klasy
    Fsc = Fsc+Ftmp; % suma Fscore dla wszystkich klas
end

[sortedFsc, index] = sort(Fsc,'descend');
sortedFeatures = features(:,index);
sortedFeatures_names = feat_names{index,:};

reducedFeaturesMat = sortedFeatures(:,1:round((p/100)*length(features)));
reducedFeaturesMat_names = sortedFeatures_names(1:round((p/100)*length(features)));
reducedFeaturesMat = [classVec, reducedFeaturesMat];
end
