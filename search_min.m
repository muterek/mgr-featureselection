function res = search_min(reducedFeaturesMat)

k_fold = 5;
n_max = size(reducedFeaturesMat,2);
step = 1000;
n_min = 1;

while step >= 10
    
    [loss, x] = FSel_results(reducedFeaturesMat,'linear',k_fold,step,n_min,n_max);
    [min_tmp, index_min_tmp] = min(loss);
    x(index_min_tmp)
    
    if n_max-n_min >= 30000
        step = 1000;
        d = 10000;
    elseif n_max-n_min <= 30000 && n_max-n_min > 10000
        step = 250;
        d = 5000;
    elseif n_max-n_min <= 10000 && n_max-n_min > 5000
        step = 100;
        d = 2500;
    elseif n_max-n_min <= 5000 && n_max-n_min > 1000
        step = 10;
        d = 500;
    else 
        break;
    end
    
    if x(index_min_tmp)-d >= 1
        n_min = x(index_min_tmp)-d;
    end
    
    if x(index_min_tmp)+d <= n_max
        n_max = x(index_min_tmp)+d;
    end
    
end

res = x(index_min_tmp);

end

