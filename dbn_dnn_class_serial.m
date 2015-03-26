function GP = dbn_dnn_class_serial(amount)

load('data/crblp_f0_class_269.mat');
% load('temp.mat');

%%
rng(0);
dbn.sizes = amount;
opts.numepochs =   20;
opts.batchsize = 10;
opts.momentum  =   0.95;
opts.alpha     =   0.0002;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%% nn for f0
nn = dbnunfoldtonn(dbn, 327);
nn.activation_function = 'sigm';
nn.output = 'softmax';
opts.numepochs = 10;
opts.batchsize = 110;
%train nn
% nn.learningRate  = 0.0002;
% nn.dropoutFraction = 0.5;
% nn.weightPenaltyL2 = 1e-4;
opts.plot               = 1; 

nn_f = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_f, bad] = nntest(nn_f, test_x, test_y);

%% nn for delta f0
nn = dbnunfoldtonn(dbn, 80);
nn.activation_function = 'sigm';
nn.output = 'softmax';
opts.numepochs = 10;
opts.batchsize = 110;
%train nn
% nn.learningRate  = 0.0002;
% nn.dropoutFraction = 0.5;
% nn.weightPenaltyL2 = 1e-4;
opts.plot               = 1; 

nn_d = nntrain(nn, train_nnx, train_nndy, opts,val_x,val_dy);
[er_d, bad] = nntest(nn_d, test_x, test_nndy);

%% nn for delta delta f0
nn = dbnunfoldtonn(dbn, 139);
nn.activation_function = 'sigm';
nn.output = 'softmax';
opts.numepochs = 10;
opts.batchsize = 110;
%train nn
% nn.learningRate  = 0.0002;
% nn.dropoutFraction = 0.5;
% nn.weightPenaltyL2 = 1e-4;
opts.plot               = 1; 

nn_dd = nntrain(nn, train_nnx, train_nnddy, opts,val_x,val_ddy);
[er_dd, bad] = nntest(nn_dd, test_x, test_nnddy);

%%
% f0 prediction as Class (330 class)
% load different nn weights and sentence_id see the graph
% although results are very bad
%clear;clc;
close all;
% load('data/delta_f0.mat');
load('data/input_269.mat');
load('data/sen_index.mat');load('data/word_index.mat');load('data/syn_duration.mat')
%%
%load('saved_weights/dbn_NN_Class_R_3L_450_99m_0002alpha.mat');
%load('temp_dbn_dnn.mat');
sentence_id = 10337;
[sen_i, ~] = find(sen_index(:,2)==sentence_id);
sen_dur = sen_index(sen_i:sen_i+1,1);
[w_i1,~] = find(word_index==sen_dur(1));
[w_i2,~] = find(word_index==sen_dur(2));
word_i = word_index(w_i1:w_i2-1);
sen_dur(2) = sen_dur(2)-1;

input_text=input((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:); 
% NN predicted f0
f0 = []; d_f0=[];dd_f0=[];
labels = nnpredict(nn_f, input_text);
for i=1:size(input_text,1)
    a=labels(i,1);
    if(a==1)
        a=0;
    else
        a=a+73;
    end
    f0 = [f0;a];
end

labels = nnpredict(nn_d, input_text);
load('data/delta_f0.mat');
z=unique(ceil(delta));
z=reshape(z,[10 80])';
for i=1:size(input_text,1)
    a=labels(i,1);
    a = median(z(a,:));
    d_f0 = [d_f0;a];
end
labels = nnpredict(nn_dd, input_text);
z=unique(ceil(delta_delta));
z=reshape(z,[8 139])';
for i=1:size(input_text,1)
    a=labels(i,1);
    a = median(z(a,:));
    dd_f0 = [dd_f0;a];
end


%% for delta and delta delta feature combination
% load('data/delta_f0.mat');
% d = delta((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:);
% dd = delta_delta((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:);
target_f0 = add_delta_deltadelta(f0,d_f0,dd_f0);
target_f0(target_f0>0)=0;
%%
% load('data/f0_5state.mat');
% o=f0_5state(sen_dur(1):sen_dur(2),:); o=reshape(o',1,[])';
load('data/f0.mat');
o=f0_5state((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:);
figure;plot(o);hold on;plot(-target_f0,'r');hold off;
rms = rmse(o,-target_f0)
xcor = xcorr(o,-target_f0,0,'coeff')
path = ['temp/dbn_NN_Class269_' num2str(amount(1)) '_' num2str(amount(2)) '_' num2str(amount(3)) '_' num2str(rms) '_' num2str(xcor) '.mat'];
save(path, 'dbn','nn_f','nn_d','nn_dd');
clear;clc;
end
