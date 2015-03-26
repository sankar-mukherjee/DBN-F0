clear;clc;
load('weights_theano/450_dbn_351.pkl.mat');

for i=0:3
   ww = ['w' num2str(i)];
   bb = ['b' num2str(i)];  
   nn.W{1,i+1} =  [eval(ww)' eval(bb)];
end

%%
load('data/input_351.mat');
load('data/sen_index.mat');load('data/word_index.mat');load('data/syn_duration.mat');

sentence_id = 10337;
[sen_i, ~] = find(sen_index(:,2)==sentence_id);
sen_dur = sen_index(sen_i:sen_i+1,1);
[w_i1,~] = find(word_index==sen_dur(1));
[w_i2,~] = find(word_index==sen_dur(2));
word_i = word_index(w_i1:w_i2-1);
sen_dur(2) = sen_dur(2)-1;

input_text=input((sen_dur(1)-1)*5+1:(sen_dur(2)-1)*5+5,:); 

f0 = [];
for i=1:size(input_text,1)
    a = NN_predict(nn,input_text(i,:));
    f0 = [f0;a];    
end

% % NN predicted f0
% f0 = []; d_f0=[];dd_f0=[];
% labels = nnpredict(nn_f, input_text);
% for i=1:size(input_text,1)
%     a=labels(i,1);
%     if(a==1)
%         a=0;
%     else
%         a=a+73;
%     end
%     f0 = [f0;a];
% end