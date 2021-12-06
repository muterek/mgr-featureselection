function [datat, data, classVec, features, data_name, feat_names] = read_file(file)

% Wejœcie:
% 
% - file - nazwa pliku z rozszerzeniem .txt
% 
% Wyjœcie:
% 
% - data - macierz ustandaryzowanych danych liczbowych, której wiersze odpowiadaj¹ próbkom,a
% kolumny - cechom. Pierwsza cecha jest klas¹, do której nale¿y dana
% próbka,
%  - classVec - wektor kolumnowy z identyfikatorami klas,
%  - features - macierz pozosta³ych cech (bez cechy z identyfikatorami klas),
%  - data_name - nazwa zestawu,
%  - feat_names - tabela zawieraj¹ca nazwy cech.
 
datat = readtable(file);
data = datat{:,2:end};

data_name_tmp = strsplit(file,'.');
data_name = data_name_tmp{1};

datat = readtable(file);
feat_names = datat(2:end,1); % nazwy cech

data=data.';   % transpozycja macierzy - w kolumnach cechy, w wierszach próbki
data(:,2:end) = zscore(data(:,2:end)); % normalizacja z-score cech (standaryzacja)
classVec = data(:,1); % wektor klas
features = data(:,2:end); % macierz cech
end 