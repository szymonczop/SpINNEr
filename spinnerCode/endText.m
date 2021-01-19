%% This function returns the ending of ordinal based on the cardinal number	 	

%%
function out = endText(ii)
iistr    = num2str(ii);
iilast   = str2num(iistr(end)); %#ok<*ST2NM>

if iilast == 1
    out = 'st';
else
    if iilast == 2
        out = 'nd';
    else
        if iilast == 3
            out = 'rd';
        else
            out = 'th';
        end
    end
end

if and(ii>3, ii<21)
    out = 'th';
end

end