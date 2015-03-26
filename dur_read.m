clear;clc;
filelist_tag = dir('dur/*.dur');
for i=1:length(filelist_tag)
    dur = ['dur/' num2str(filelist_tag(i,1).name)];
    dur = importdata(dur);
    dur(cellfun(@isempty,dur)) = [];
    dur = regexp(dur, '([^ \t][^\t]*)', 'match');
    total_dur=[];
    for k=1:size(dur,1)
       if(6 * round(k/6) ~= k)
           total_dur = [total_dur; dur{k,1}];
       end
    end
    clear dur;
    for j=1:length(total_dur)
        R_dur{i,j} = str2num(total_dur{j,1}(strfind(total_dur{j,1},'duration')+9:strfind(total_dur{j,1},'(frame)')-1));        
    end
    dur_name{i,:} = strrep(num2str(filelist_tag(i,1).name),'.dur','');
end

save('data/syn_duration.mat', 'R_dur', 'dur_name')