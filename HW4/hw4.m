%test 1
clear; close all; clc

% artist 1: sodagreen(first 6 songs)
% artist 2: (g)-idle (middle 6 songs)
% artist 3: KUN (last 6 songs)
filename = ['miss.mp3';'rain.mp3';'meet.mp3';'walk.mp3';'lone.mp3';'song.mp3';
            'blow.mp3';'lata.mp3';'seno.mp3';'hann.mp3';'lion.mp3';'uhoh.mp3';
            'yong.mp3';'rebi.mp3';'pull.mp3';'iwgl.mp3';'snow.mp3';'wait.mp3'];

test = [];
sec = [10 20 30 40 50 60 70 80 90 100 110];
for k = 1:size(filename,1)
   [y,Fs] = audioread(filename(k,:));
    y = y(:,1); 
    for i = 1:length(sec)
        p = y(sec(i)*Fs:Fs*(5+sec(i))); p=p';
        Sp =abs(spectrogram(p));
        Sp = reshape(Sp, [1,8*32769]);
        test = [test; Sp];
    end
end        

%%
test = test';
feature=26; % 1<feature<length(sec)*size(filename,1)
[U,S,V,w,sort1,sort2,sort3] = dc_trainer(test,feature);

% plot our projected data and threshold
plot(sort1,1,'go'),hold on
plot(sort2,2,'ko')
plot(sort3,3,'bo')
xlabel('projection onto w'), ylabel('distinguishable values')
title('Test1 traning data')
set(gca,'Fontsize',16) 
%% create threshold based on previous graph
[t1,t2] = threshold(sort2,sort3,sort1);
xline(t1,'--');
xline(t2,'-');

%% Test classifier
filename2 = ['SMAL.mp3';'NAME.mp3';'HTG_.mp3'];
TestSet = [];
sec = [10 20 30 40 50 60 70 80 90 100 110];
for k = 1:size(filename2,1)
    [y,Fs] = audioread(filename2(k,:));
    y = y(:,1); 
    for i = 1:length(sec)
        p = y(sec(i)*Fs:Fs*(5+sec(i))); p=p';
        Sp =abs(spectrogram(p));
        Sp = reshape(Sp, [1,8*32769]);
        TestSet = [TestSet; Sp];
    end
end

TestSet = TestSet';
%%
label1 = ones(1,length(sec));
label2 = ones(1,length(sec)) * 2;
label3 = ones(1,length(sec)) * 3;
hiddenlabels = [label1, label2, label3];

TestMat = U'*TestSet;  % PCA projection
pval = w'*TestMat;  % LDA projection

%% Let's find out accuracy!

r = [];
for i = 1:(size(filename2,1)*length(sec))
    if pval(i)<=t1
        r = [r; 2];
    elseif pval(i) <=t2
        r = [r; 3];
    else
        r = [r; 1];
    end
end
r = r';

disp('Number of mistakes')
errNum = sum(abs(r-hiddenlabels)~=0)

disp('Rate of success');
sucRate = 1-errNum/(size(filename2,1)*length(sec))
%%
% [~,~,~,~,sortA,sortB,sortC] = dc_trainer(TestSet,26);

    plot(pval(1:11),1,'m*'),hold on
    plot(pval(12:26),2,'c*')
    plot(pval(27:end),3,'r*')
    
    title('Test1 traning data with thresholds and test data')
    
%%
function [threshold1,threshold2] = threshold(sort1,sort2,sort3)
    t1 = length(sort1);
    t2 = 1;
    while sort1(t1)>sort2(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold1 = (sort1(t1)+sort2(t2))/2;

    t1 = length(sort2);
    t2 = 1;
    while sort2(t1)>sort3(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold2 = (sort2(t1)+sort3(t2))/2;
end


function [U,S,V,w,sort1,sort2,sort3] = dc_trainer(test,feature)
    n1 = size(test,2)/3; n2 = size(test,2)/3;n3 = size(test,2)/3;
    
    [U,S,V] = svd(test,'econ');
    
    bands = S*V'; % projection onto principal components
    U = U(:,1:feature);
    
    band1 = bands(1:feature,1:n1);
    band2 = bands(1:feature,n1+1:n1+n2);
    band3 = bands(1:feature,n1+n2+1:n1+n2+n3);
    
    m1 = mean(band1,2);
    m2 = mean(band2,2);
    m3 = mean(band3,2);
    totalm = mean(bands,2);  
    totalm = totalm(1:feature,:);% why?
    
    Sw = 0; % within class variances
    for k=1:n1
        Sw = Sw + (band1(:,k)-m1)*(band1(:,k)-m1)';
    end
    for k=1:n2
        Sw = Sw + (band2(:,k)-m2)*(band2(:,k)-m2)';
    end
    for k=1:n3
        Sw = Sw + (band3(:,k)-m3)*(band3(:,k)-m3)';
    end
    
     % between class 
    Sb = (m1-totalm)*(m1-totalm)'+(m2-totalm)*(m2-totalm)'+(m3-totalm)*(m3-totalm)';
    
    [V2,D] = eig(Sb,Sw); % linear discriminant analysis
    [~,ind] = max(abs(diag(D)));
    w = V2(:,ind); w = w/norm(w,2);
    
    v1 = w'*band1; 
    v2 = w'*band2;
    v3 = w'*band3;
    
    sort1 = sort(v1);
    sort2 = sort(v2);
    sort3 = sort(v3);
    
%     plot(sort1,1,'go'),hold on
%     plot(sort2,2,'ko')
%     plot(sort3,3,'bo')

end

