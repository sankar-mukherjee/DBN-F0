clear;clc;
load('data/crblp_f0.mat');%load('temp.mat');
%%
rng(0);
dbn.sizes = [100 100 100];
opts.numepochs =   20;
opts.batchsize = 10;
opts.momentum  =   0.95;
opts.alpha     =   0.0002;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%%
nn = dbnunfoldtonn(dbn, 1);
nn.activation_function = 'sigm';
nn.output = 'sigm';
opts.numepochs = 30;
opts.batchsize = 10;
%train nn
nn.learningRate  = 0.0002;
nn.dropoutFraction = 0.5;
nn.weightPenaltyL2 = 1e-4;
opts.plot               = 1; 

train_nndy=train_nny(:,2); val_dy=val_y(:,2);test_dy=test_y(:,2);
train_nnddy=train_nny(:,3);val_ddy=val_y(:,3);test_ddy=test_y(:,3);
train_nny=train_nny(:,1);val_y=val_y(:,1);test_y=test_y(:,1);
nn_f = nntrain(nn, train_nnx, train_nny, opts,val_x,val_y);
[er_f, bad] = nntest(nn_f, test_x, test_y);

nn_d = nntrain(nn, train_nnx, train_nndy, opts,val_x,val_dy);
[er_d, bad] = nntest(nn_d, test_x, test_dy);

nn_dd = nntrain(nn, train_nnx, train_nnddy, opts,val_x,val_ddy);
[er_dd, bad] = nntest(nn_dd, test_x, test_ddy);