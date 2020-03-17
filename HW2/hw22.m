clear; close all; clc

[y1,Fs] = audioread('music1.wav');
v1 = y1';
tr_piano=length(y1)/Fs; % record time in seconds

subplot(2,1,1)
plot((1:length(y1))/Fs,y1);
hold on
xlabel('Time [sec]'); ylabel('Amplitude');
title('Mary had a little lamb (piano)');

L=tr_piano; n=701440; 
t2=linspace(0,L,n+1); t=t2(1:n); 
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
vt = fft(v1);

subplot(2,1,2)
plot(ks,fftshift(abs(vt)))
xlabel('FFT(Sg)'),ylabel('frequency')

%% 
% a decrease, width increase
% delta t big == less time resolution; delta t small, more time resolution

figure(2)
a = 100;
tslide=0:0.1:L;
Sgt_spec = zeros(length(tslide),n);
Sgt_spec = [];
for j=1:length(tslide)
    g=exp(-a*(t-tslide(j)).^2); 
    Sg=g.*v1; 
    Sgt=fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt)); % We don't want to scale it
end

pcolor(tslide,ks./(2*pi),Sgt_spec.')
title({'Piano Spectrogram','a=100, \Deltat=0.1'})
xlabel('Time(s)'),ylabel('Frequency(Hz)')
shading interp 
set(gca,'Ylim',[230 370],'Fontsize',16) 
colormap(hot)




