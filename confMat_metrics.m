function metricsTable = confMat_metrics(cTest_matrix,predictedLabels_matrix)
J = unique(cTest_matrix); % classes

tp = zeros(1,length(J));
fp = zeros(1,length(J));
tn = zeros(1,length(J));
fn = zeros(1,length(J));

for k=1:length(J)
    tp(k) = sum((predictedLabels_matrix == k) & (cTest_matrix == k));
    fp(k) = sum((predictedLabels_matrix == k) & (cTest_matrix ~= k));
    tn(k) = sum((predictedLabels_matrix ~= k) & (cTest_matrix ~= k));
    fn(k) = sum((predictedLabels_matrix ~= k) & (cTest_matrix == k));
end

ACC = (tp+tn)./(tp+fp+tn+fn);
Error_rate = 1-ACC;
czulosc = tp./(tp+fn);
specyficznosc = tn./(fp+tn);
precyzja = tp./(tp+fp);
F1_score = (2.*precyzja.*czulosc)./(precyzja+czulosc);
FPR = 1-specyficznosc;
FNR = fn./(fn+tp);

single_metrics = [ACC',Error_rate',czulosc', specyficznosc', precyzja', F1_score', FPR', FNR'];

ACC_macro = sum(ACC)/length(J);
Error_rate_macro = sum(Error_rate)/length(J);
czulosc_macro = sum(czulosc)/length(J);
specyficznosc_macro = sum(specyficznosc)/length(J);
precyzja_macro = sum(precyzja)/length(J);
F1_score_macro = sum(F1_score)/length(J);
FPR_macro = sum(FPR)/length(J);
FNR_macro = sum(FNR)/length(J);

macro_metrics = [ACC_macro, Error_rate_macro, czulosc_macro, specyficznosc_macro, precyzja_macro, F1_score_macro, FPR_macro, FNR_macro];

czulosc_micro = sum(tp)/sum(tp+fn);
specyficznosc_micro = sum(tn)/sum(fp+tn);
precyzja_micro = sum(tp)/sum(tp+fp);
F1_score_micro = (2.*precyzja_micro.*czulosc_micro)/sum(precyzja_micro+czulosc_micro);
FPR_micro = 1-specyficznosc_micro;
FNR_micro = sum(fn)/sum(fn+tp);

micro_metrics = [NaN, NaN, czulosc_micro, specyficznosc_micro, precyzja_micro, F1_score_micro, FPR_micro, FNR_micro];

table_val = [single_metrics;macro_metrics;micro_metrics];
metricsTable = table();
metricsTable.Variables = mat2cell(table_val,ones(size(table_val,1),1),ones(size(table_val,2),1));
metricsTable.Properties.VariableNames = {'dokładność','błąd','czułość','specyficzność','precyzja','miara F1','FPR','FNR'};
metricsTable.Properties.RowNames = [cellstr(string(J));{'M'};{'u'}];

end

