load handel
v = y';
plot((1:length(v))/Fs,v);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Signal of Interest, v(n)');

n=73113; L=n./Fs;
t2=linspace(0,L,n+1); t=t2(1:n); 
k=(2*pi/L)*[0:(n-1)/2 -(n-1)/2:-1]; 
ks=fftshift(k);
vt = fft(v);


%% plot spectrogram of gabor transform of v
figure(1)
a_vec = [0.1 60 100 10000];
tslide=0:0.1:L;
for jj = 1:length(a_vec)
    a = a_vec(jj);
    Sgt_spec = zeros(length(tslide),n);
    Sgt_spec = [];
    for j=1:length(tslide)
        g=exp(-a*(t-tslide(j)).^2); 
        Sg=g.*v; 
        Sgt=fft(Sg); 
        Sgt_spec(j,:) = fftshift(abs(Sgt)); 
    end

    subplot(2,2,jj)
    pcolor(tslide,ks,Sgt_spec.'), 
    shading interp 
    title(['a = ',num2str(a)],'Fontsize',16)
    xlabel('Time(s)'),ylabel('Frequency(Hz)')
    set(gca,'Fontsize',16) 
    colormap(hot)
end
sgtitle('Gaussian filter','Linewidth',18)

%% Mexican hat wavelet

figure(2)
a_vec = [0.001 0.01 0.1 10];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.1:L;
    Sgt_spec = zeros(length(tslide),n);
    Sgt_spec = [];
    for j=1:length(tslide)
        m = 1/sqrt(a).*(1- ( (t-tslide(j)) ./a ).^2).*exp(-( (t-tslide(j)) ./a).^2./2);
        Sg=m.*v; 
        Sgt=fft(Sg); 
        Sgt_spec(j,:) = fftshift(abs(Sgt)); 
    end
    
    subplot(2,2,jj)
    pcolor(tslide,ks./(2*pi),Sgt_spec.'), 
    shading interp 
    title(['a = ',num2str(a)],'Fontsize',16)
    xlabel('Time(s)'),ylabel('Frequency(Hz)')
    set(gca,'Fontsize',16) 
    colormap(hot)
end
sgtitle('Mexican hat wavelet','Linewidth',16)

%% step function

figure(3)
a_vec = [0.001 0.01 0.1 10];
for jj = 1:length(a_vec)
    a = a_vec(jj);
    tslide=0:0.1:L;
    Sgt_spec = zeros(length(tslide),n);
    Sgt_spec = [];
    for j=1:length(tslide)
        s = abs(t-tslide(j))<=a;
        Sg=s.*v; 
        Sgt=fft(Sg); 
        Sgt_spec(j,:) = fftshift(abs(Sgt)); 
    end
    
    subplot(2,2,jj)
    pcolor(tslide,ks./(2*pi),Sgt_spec.'), 
    shading interp 
    title(['a = ',num2str(a)],'Fontsize',16)
    xlabel('Time(s)'),ylabel('Frequency(Hz)')
    set(gca,'Fontsize',16) 
    colormap(hot)
end
sgtitle('Shannon Wavelet','Linewidth',16)



