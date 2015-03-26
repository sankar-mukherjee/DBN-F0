diary ('dbn_7L_80_0.9m_0.05_0.005alpha_error_report');


%%
clear;
load('data/input.mat');
train_x = input(160247:end,:);

rng(0);
dbn.sizes = [80 80 80 80 80 80 80];
opts.numepochs =   50;
opts.batchsize = 100;
opts.momentum  =   0.9;
opts.alpha     =   0.05;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);          save('saved_weights/dbn_7L_80_0.9m_0.05alpha.mat','dbn')

%%
clear;
load('data/input.mat');
train_x = input(160247:end,:);

rng(0);
dbn.sizes = [80 80 80 80 80 80 80];
opts.numepochs =   50;
opts.batchsize = 100;
opts.momentum  =   0.9;
opts.alpha     =   0.005;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);          save('saved_weights/dbn_4L_80_0.9m_0.005alpha.mat','dbn')

diary ('off');