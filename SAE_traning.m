diary ('SAE_100_80_sigm_softrelu_error_report');
clear;clc;
load('data/input.mat');
train_x = input(160247:end,:);

%% ex2 train a 100 hidden unit SDAE
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([220 100 100 100 100 100 100 100]);

for i=1:length(sae.ae)
    sae.ae{i}.activation_function       = 'sigm';
    sae.ae{i}.learningRate              = 0.4;
    sae.ae{i}.inputZeroMaskedFraction   = 0.5;
end

opts.numepochs = 100;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);

%add pretrained weights
for i=1:length(sae.ae)
    nn.W{i} = sae.ae{i}.W{1};
end
save('saved_weights/SAE_7L_100_4lrn_5zmask_sigm.mat','nn')

%% ex2 train a 80 hidden unit SDAE
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([220 80 80 80 80 80 80 80]);

for i=1:length(sae.ae)
    sae.ae{i}.activation_function       = 'sigm';
    sae.ae{i}.learningRate              = 0.4;
    sae.ae{i}.inputZeroMaskedFraction   = 0.5;
end

opts.numepochs = 100;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);

%add pretrained weights
for i=1:length(sae.ae)
    nn.W{i} = sae.ae{i}.W{1};
end
save('saved_weights/SAE_7L_80_4lrn_5zmask_sigm.mat','nn')

%% ex2 train a 80 hidden unit SDAE
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([220 80 80 80 80 80 80 80]);

for i=1:length(sae.ae)
    sae.ae{i}.activation_function       = 'softrelu';
    sae.ae{i}.learningRate              = 0.4;
    sae.ae{i}.inputZeroMaskedFraction   = 0.5;
end

opts.numepochs = 100;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);

%add pretrained weights
for i=1:length(sae.ae)
    nn.W{i} = sae.ae{i}.W{1};
end
save('saved_weights/SAE_7L_80_4lrn_5zmask_sigm_softrelu.mat','nn')
%% ex2 train a 100 hidden unit SDAE
%  Setup and train a stacked denoising autoencoder (SDAE)
rng(0);
sae = saesetup([220 100 100 100 100 100 100 100]);

for i=1:length(sae.ae)
    sae.ae{i}.activation_function       = 'softrelu';
    sae.ae{i}.learningRate              = 0.4;
    sae.ae{i}.inputZeroMaskedFraction   = 0.5;
end

opts.numepochs = 100;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);

%add pretrained weights
for i=1:length(sae.ae)
    nn.W{i} = sae.ae{i}.W{1};
end
save('saved_weights/SAE_7L_100_4lrn_5zmask_softrelu.mat','nn')

diary('off');
