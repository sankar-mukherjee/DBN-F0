clear;clc;
load('data/input_351.mat');

%% for output class
load('data/f0_class.mat');load('data/delta_f0_class.mat');


%% remove the zero o/p samples accroding to f0_5state. 1712651 no of nonzero element in f0_5state
f0=f0_5state;
df0=delta;
ddf0=delta_delta;
inp=input;

f0_5state = zeros(1712651,327);
delta = zeros(1712651,80);
delta_delta = zeros(1712651,139);
input = zeros(1712651,351);
c=1;
for i=1:size(inp,1)
   if(f0(i,1)~=1)
       f0_5state(c,:) = f0(i,:);
       delta(c,:) = df0(i,:);
       delta_delta(c,:) = ddf0(i,:);
       input(c,:) = inp(i,:);
       c=c+1;
   end
end

clear f0 df0 ddf0 inp c i;
%%
traning_size = size(input,1);

n_train = traning_size *60/100;
nn_train = traning_size *20/100;
n_valid = floor(traning_size *15/100);
n_test = floor(traning_size *5/100);

n_train = floor(n_train/10)*10;
nn_train = floor(nn_train/10)*10;

total_input = 1:traning_size;
[n_train i]=datasample(total_input,n_train,'Replace',false);    total_input(i) = [];
[nn_train i]=datasample(total_input,nn_train,'Replace',false);    total_input(i) = [];
[n_valid i]=datasample(total_input,n_valid,'Replace',false);    total_input(i) = [];
[n_test i]=datasample(total_input,n_test,'Replace',false);    total_input(i) = [];

train_x = input(n_train,:);
% train_y = f0_5state(n_train,:);           % not need for DBN or NN
% train_dy = delta(n_train,:);    
% train_ddy = delta_delta(n_train,:);

train_nnx = input(nn_train,:);
train_nny = f0_5state(nn_train,:); 
train_nndy = delta(nn_train,:);    
train_nnddy = delta_delta(nn_train,:); 

val_x = input(n_valid,:);
val_y = f0_5state(n_valid,:);
val_dy = delta(n_valid,:);    
val_ddy = delta_delta(n_valid,:);

test_x = input(n_test,:);
test_y = f0_5state(n_test,:);
test_nndy = delta(n_test,:);    
test_nnddy = delta_delta(n_test,:); 

clear n_train nn_train n_valid n_test input f0_5state i delta delta_delta traning_size total_input;
