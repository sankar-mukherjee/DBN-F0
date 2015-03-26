diary ('dbn_nn2152_6016_80_100_error_report');

clear;
load('data/f0_5state.mat');load('data/input_2152.mat');
traning_size = 160246;
f0_5state = reshape(f0_5state',1,[])';
N_f0 = norm(f0_5state);
n_train = ceil((traning_size * 3 / 4)/100)*100;
n_valid = floor((traning_size /4)/100)*100;

train_xx = input(1:n_train,:);
train_yy = f0_5state(1:n_train*5,:) / N_f0;

val_x = input(n_train+1:n_train+n_valid,:);
val_y = f0_5state((n_train*5)+1:(n_train+n_valid)*5,:)/ N_f0;

train_xx = expand(train_xx,[5,1]);
val_x = expand(val_x,[5,1]);

%%
load('saved_weights/dbn2152_7L_80_5m_6alpha.mat');disp('dbn2152_7L_80_5m_6alpha')

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'sigm';
nn.output = 'linear';
%train nn
opts.numepochs = 10;
opts.batchsize = 100;
nn = nntrain(nn, train_xx, train_yy, opts,val_x,val_y); save('saved_weights/dbn2152_NN_7L_80_5m_6alpha.mat','nn')
%%
load('saved_weights/dbn2152_7L_100_5m_6alpha.mat');disp('dbn2152_7L_100_5m_6alpha')

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'sigm';
nn.output = 'linear';
%train nn
opts.numepochs = 10;
opts.batchsize = 100;
nn = nntrain(nn, train_xx, train_yy, opts,val_x,val_y);save('saved_weights/dbn2152_NN_7L_100_5m_6alpha.mat','nn')
%%

clear;
load('data/f0_5state.mat');load('data/input_6016.mat');
traning_size = 160246;
f0_5state = reshape(f0_5state',1,[])';
N_f0 = norm(f0_5state);
n_train = ceil((traning_size * 3 / 4)/100)*100;
n_valid = floor((traning_size /4)/100)*100;

train_xx = input(1:n_train,:);
train_yy = f0_5state(1:n_train*5,:) / N_f0;

val_x = input(n_train+1:n_train+n_valid,:);
val_y = f0_5state((n_train*5)+1:(n_train+n_valid)*5,:)/ N_f0;

train_xx = expand(train_xx,[5,1]);
val_x = expand(val_x,[5,1]);

load('saved_weights/dbn6016_7L_80_5m_6alpha.mat');disp('dbn6016_7L_80_5m_6alpha')

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'sigm';
nn.output = 'linear';
%train nn
opts.numepochs = 10;
opts.batchsize = 100;
nn = nntrain(nn, train_xx, train_yy, opts,val_x,val_y);save('saved_weights/dbn6016_NN_7L_80_5m_6alpha.mat','nn')
%%
load('saved_weights/dbn6016_7L_100_5m_6alpha.mat');disp('dbn6016_7L_100_5m_6alpha')

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'sigm';
nn.output = 'linear';
%train nn
opts.numepochs = 10;
opts.batchsize = 100;
nn = nntrain(nn, train_xx, train_yy, opts,val_x,val_y);save('saved_weights/dbn6016_NN_7L_100_5m_6alpha.mat','nn')

diary ('off');
