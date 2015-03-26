% data preprocess for regression models

load('data/input_269.mat');
%% for output real value
load('data/f0.mat');load('data/delta_f0.mat');

traning_size = size(f0_5state,1);
%% scaling data between 0.1 0.99
range2 = 0.99 - 0.1;

f0 = f0_5state;
f0_5state=(((f0-min(f0(:)))/(max(f0(:)) -min(f0(:))))*range2) + 0.1;
scaling_factors(1,1)= (max(f0(:)) -min(f0(:))); scaling_factors(1,2) = min(f0(:));

f0 = delta;
delta=(((f0-min(f0(:)))/(max(f0(:)) -min(f0(:))))*range2) + 0.1;
scaling_factors(2,1)= (max(f0(:)) -min(f0(:))); scaling_factors(2,2) = min(f0(:));

f0 = delta_delta;
delta_delta=(((f0-min(f0(:)))/(max(f0(:)) -min(f0(:))))*range2) + 0.1;
scaling_factors(3,1)= (max(f0(:)) -min(f0(:))); scaling_factors(3,2) = min(f0(:));

f0_5state = [f0_5state delta delta_delta]; 
%% random suffling
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

val_x = input(n_valid,:);
val_y = f0_5state(n_valid,:);

test_x = input(n_test,:);
test_y = f0_5state(n_test,:);

clear n_train nn_train n_valid n_test input f0_5state i total_input traning_size delta delta_delta f0;