clear;clc;
load('data/crblp_f0_class_269.mat');
% load('temp.mat');

%%
rng(0);
dbn.sizes = [200 200 200];
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