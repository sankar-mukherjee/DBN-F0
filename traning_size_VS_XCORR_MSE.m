%%for ICASSP-2014 traning_labelled_data VS XCORR and RMSE
clear;clc;
% dbn = dbn(10000);
load('../data/sen_index.mat');load('../data/input_6016.mat');load('../data/f0.mat');
% load('dbn.mat');load('dbn-nn-15k.mat');
load('../data/NN/dbn_nn_6016_50_30p.mat');
result_xcorr = [];result_mse = [];
Test_sample_sentence = 5;

for ii=1:1
        traning_size = floor(460246*(3/10))
     traning_size = 2000;
    
    %% NN
%     n_train = ceil((traning_size * 3 / 4)/100)*100;
%     n_valid = floor((traning_size /4)/100)*100;
% 
%     train_xx = input(1:n_train,:);
%     train_yy = f0(1:n_train,:) / 399.9851;
%     
%     val_x = input(n_train:n_train+n_valid,:);
%     val_y = f0(n_train:n_train+n_valid,:)/ 399.9851;
%     %unfold dbn to nn
%     nn = dbnunfoldtonn(dbn, 3);
%     nn.activation_function = 'sigm';
%     nn.output = 'linear';
%     %train nn
%     opts.numepochs = 50;
%     opts.batchsize = 100;
%     nn = nntrain(nn, train_xx, train_yy, opts,val_x,val_y);

    %% TGP train
    
    train_x = input(1:traning_size,:);    train_y = f0(1:traning_size,:);
%     [InvIK, InvOK, Param]= TGP(train_x,train_y,nn);
    
    %%
    
    
    train_nn_x = [];
    for i=1:size(train_x,1)
        train_nn = [1 train_x(i,:)];
        train_nn = sigm(train_nn * nn.W{1,1}');
        train_nn = [1 train_nn];
        train_nn = sigm(train_nn * nn.W{1,2}');
        train_nn = [1 train_nn];
        train_nn_x =[train_nn_x ;sigm(train_nn * nn.W{1,3}')];
    end
   XCOR=[];   MSE = [];MSE_TGP=[];MSE_GP=[];LOG=[];
   for s=1:Test_sample_sentence      
       sen_i = 9200+s;
       sen_dur = sen_index(sen_i:sen_i+1,1);       
       sen_dur(2) = sen_dur(2)-1;
       input_text=input(sen_dur(1):sen_dur(2),:);  
       target_f0 = [];
       test_nn_x =[];
       %test data with NN
       for i=1:size(input_text,1)
           train_nn = [1 input_text(i,:)];
           train_nn = sigm(train_nn * nn.W{1,1}');
           train_nn = [1 train_nn];
           train_nn = sigm(train_nn * nn.W{1,2}');
           train_nn = [1 train_nn];
           test_nn_x =[test_nn_x ;sigm(train_nn * nn.W{1,3}')];
       end
       
       %% GPML
       
       covfunc = @covSEiso;
%        nu = fix(size(train_nn_x,1)/2); iu = randperm(size(train_nn_x,1)); iu = iu(1:nu); u = train_nn_x(iu,:);
nu = fix(size(train_x,1)/2); iu = randperm(size(train_x,1)); iu = iu(1:nu); u = train_x(iu,:);  %RAW
       covfuncF = {@covFITC, {covfunc},u};
       likfunc = @likGauss;
       sn = 0.1;
       hyp2.cov = [0 ; 0];
       hyp2.lik = log(sn);
       predicted_f0=[];
       loglike = [];
       for i=1:size(train_y,2)
           % hyp2 = minimize(hyp2, @gp, -100, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i));
           hyp2 = minimize(hyp2, @gp, -100, @infFITC, [], covfuncF, likfunc, train_x, train_y(:,i));    %RAW
           exp(hyp2.lik)
           %            nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i))
           %            [m ~] = gp(hyp2, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i), test_nn_x);
           nlml2 = gp(hyp2, @infFITC, [], covfuncF, likfunc, train_nn_x, train_y(:,i))             %RAW
           [m ~] = gp(hyp2, @infFITC, [], covfuncF, likfunc, train_x, train_y(:,i), input_text);  %RAW
           predicted_f0=[predicted_f0 m];
           loglike = [loglike nlml2];
       end
       original_f0 = f0(sen_dur(1):sen_dur(2),:); 
       [Error, TGPErrorvec] = JointError(predicted_f0, original_f0);      MSE_GP = [MSE_GP;Error];       LOG = [LOG mean(loglike)];
              
%        
       
       %% TGP
%        predicted_f0 = (TGPTest(test_nn_x, train_nn_x, train_y, Param, InvIK, InvOK));
%        predicted_f0 = (TGPTest(input_text, train_x, train_y, Param, InvIK, InvOK));     %RAW Features
       
       %%
       %        target_f0 = target_f0.*399.9851;                   %predicted F0
       original_f0 = f0(sen_dur(1):sen_dur(2),:);          %original F0
       %        p = reshape(target_f0',1,[]);
       
       p = reshape(predicted_f0',1,[]);
       o = reshape(original_f0',1,[]);
       %% Error calculation
       [Error, TGPErrorvec] = JointError(predicted_f0, original_f0);      MSE_TGP = [MSE_TGP;Error];
       
       xcor = xcorr(o,p,0,'coeff');       mse = rmse(o,p);
       XCOR = [XCOR;xcor];      MSE = [MSE;mse];
   end
   
   result_xcorr = [result_xcorr; mean(XCOR)];result_mse = [result_mse ;mean(MSE)];
end