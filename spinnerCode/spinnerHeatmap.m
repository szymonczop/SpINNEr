function spinnerHeatmap(b, tit, bottom, top)
if nargin == 1
    tit ='';
end

if nargin <= 2
    b       = double(b);
    bottom  = min(min(b));
    top     = max(max(b));
end

%% heatmap
margins = [0.05, 0.05];
figure('name','Estimates');
axes('Position', [margins(2), margins(1), (1-2*margins(2)), (1-2*margins(1))]);
imagesc(b);
colormap(spinnerColormap(bottom ,top));
title(tit);
axis equal;
axis tight;
caxis manual
caxis([bottom  top]);
colorbar

end
