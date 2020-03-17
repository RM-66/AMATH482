clear; close all; clc;
load Testdata

L=15; % spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks); % Kx, Ky, Kz are all 64*64*64 size

ave = 0;
for j=1:20
    Un(:,:,:)=reshape(Undata(j,:),n,n,n);  
    ave = ave + fftn(Un);
end
ave = abs(fftshift(ave)/20);

isosurface(Kx,Ky,Kz,ave/max(max(max(ave))),0.7);
hold on
axis([-8 8 -8 8 -8 8]), grid on, drawnow
xlabel('Kx')
ylabel('Ky')
zlabel('Kz')
title('central frequency of the marble')
set(gca,'Fontsize',15)

%%
% we observed that the central frequency is (2.094,-1.047,0)
% now we consider applying the filter function around the central frequency
close all;
tau = [2.094 -1.047 0]; % central frequency
a = 1; % width of the filter

% calculation of frequency vector in 3D
g = exp(-a*((Kx-tau(1)).^2+ (Ky-tau(2)).^2+ (Kz-tau(3)).^2)); 

for j=1:20
    Un(:,:,:)=reshape(Undata(j,:),n,n,n);  
    Sg = fftshift(g).*fftn(Un);
    Sgt = ifftn(Sg);
    Sgt = abs(Sgt)./max(max(max(abs(Sgt))));
    isosurface(X,Y,Z,Sgt,0.8),hold on
    axis([-15 15 -15 15 -15 15]), grid on, drawnow
end
xlabel('x')
ylabel('y')
zlabel('z')
title('marble signal in the time domain')
set(gca,'Fontsize',15)

%%

close all;
position = 0;
x1 = zeros(20);
y1=zeros(20);
z1=zeros(20);
for j=1:20
    Un(:,:,:)=reshape(Undata(j,:),n,n,n);  
    Sg = fftshift(g).*fftn(Un);
    Sgt = ifftn(Sg);
    Sgt = abs(Sgt)/max(max(max(abs(Sgt))));
    [M, I] = max(Sgt(:)); 
    x1(j) = X(I);
    y1(j) = Y(I);
    z1(j) = Z(I);
end
plot3(x1,y1,z1,'bo-'), hold on
axis([-15 15 -15 15 -15 15]), grid on, drawnow
xlabel('x')
ylabel('y')
zlabel('z')
title('path of marble')
set(gca,'Fontsize',15)

