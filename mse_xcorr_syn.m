% load different nn weights and sentence_id see the graph
% although results are very bad
load('data/input_class.mat');load('data/sen_index.mat');load('data/word_index.mat');load('data/syn_duration.mat')
%%
%load('saved_weights/dbn_NN_Class_R_3L_450_99m_0002alpha.mat');
 load('temp_dbn_dnn.mat');
sentence_id = 10337;
[sen_i, ~] = find(sen_index(:,2)==sentence_id);
sen_dur = sen_index(sen_i:sen_i+1,1);
[w_i1,~] = find(word_index==sen_dur(1));
[w_i2,~] = find(word_index==sen_dur(2));
word_i = word_index(w_i1:w_i2-1);
sen_dur(2) = sen_dur(2)-1;
% NN predicted f0
input_text=input((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:);  
f0 = []; d_f0=[];dd_f0=[];
% f0 = NN_predict(nn_f, input_text);
% d_f0 = NN_predict(nn_d, input_text);
% dd_f0 = NN_predict(nn_dd, input_text);
for i=1:size(input_text,1)
    a = NN_predict(nn_f,input_text(i,:));
    f0 = [f0;a];
    a = NN_predict(nn_d,input_text(i,:));
    d_f0 = [d_f0;a];
    a = NN_predict(nn_dd,input_text(i,:));
    dd_f0 = [dd_f0;a];
end
load('scaling_factor.mat');

f0 = (f0 -0.1)/(0.99 - 0.1);d_f0 = (d_f0 -0.1)/(0.99 - 0.1);dd_f0 = (dd_f0 -0.1)/(0.99 - 0.1);

f0 = f0*scaling_factors(1,1)+scaling_factors(1,2);
d_f0 = d_f0*scaling_factors(2,1)+scaling_factors(2,2);
dd_f0 = dd_f0*scaling_factors(3,1)+scaling_factors(3,2);
%% for delta and delta delta feature combination
target_f0 = add_delta_deltadelta(f0,d_f0,dd_f0);
target_f0(target_f0>0)=0;
%%
% load('data/f0_5state.mat');
% o=f0_5state(sen_dur(1):sen_dur(2),:); o=reshape(o',1,[])';
load('data/f0.mat');
o=f0_5state((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:);
figure;plot(o);hold on;plot(-target_f0,'r');hold off;
rmse(o,-target_f0)
xcor = xcorr(o,-target_f0,0,'coeff')
