% TP regression logistique STAP 2020
% Ce script contient de quoi d閙arrer le TP et des fonctions (en fin de
% script) pour repr閟enter les donn閑s et les solutions trouv閑s. 
% 
% Une partie du TP consiste � 閏rire les grandes 閠apes n閏essaires � 
% l'apprentissage. 
% Rappel : en matlab, les fonctions se mettent en fin de script. 
% Le script est alors organiser comme il suit: 
% - une partie avec les exp閞iences et commandes 
% - en fin de script, les fonctions 閏rites pour faire les
% exp閞iences. 
%
% Autre solution: cr閑r un fichier par fonction (le nom du fichier est le
% nom de la fonction)
clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chargement des donn閑s 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load reglog_data_3.mat
% Vous disposez d閟ormais de X et C 
% Regarder les donn閑s : contenu, dimensions, ... 
% On peut aussi les repr閟enter sur une figure:
% figure(1);
% plotdata(X,C)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ecriture pas � pas d'une 閜oque d'apprentissage:
% Nous  allons consid閞er l'ensemble des donn閑s d'apprentissage: 
% soit le couple X et C. 
% Vous trouverez plus loin des lignes de codes comment閑s 
% avec des "...", que vous pouvez d閏ommenter et terminer. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialisation des param鑤res: 
rng(1);
w0 = -5;
w = randn(1,2);
% Choix du pas d'apprentissage (ou learning rate): 
lr = 0.01;
% repr閟enter les donn閑s et la droite
% figure(2); % pour obtenir une nouvelle figure
% plotdata(X,C,w0,w)


n=length(C);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Boucle d'apprentissage et monitoring: 
% Nous pouvons maintenant mettre en place la boucle d'apprentissage
% Le nombre d'閜oque d'apprentissage est fix閑 par une variable. 
% L'objectif est d'observer l'関olution de certaines grandeurs 
% au cours de l'apprentissage, en particulier 
% l'関olution de la fonction de perte que l'on cherche � minimiser. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nepochs = 1000 ;
Losses = zeros(1,Nepochs);
L=Losses;
lr = 0.02;
% initialisation des param鑤res: 
rng(1);
b = -5;

w = ones(1,2);
w(1)=8;
w(2)=9;
taux_err = zeros(1,Nepochs);
norme = zeros(1,Nepochs);

for e = 1:1:Nepochs
    
    norme(e) = sqrt(b^2+w*w');
    
    Y = zeros(1,n);
    for i = 1:n
            Y(i) = sigmoid(-b^2 + (w(1)-X(1,i)).^2 + (w(2)-X(2,i)).^2);
    end
    
    taux_err(e) = sum(abs((Y>0.5)-C));
     
    Losses(e) = 0;
    for i = 1:n
        if C(i) == 1
            Losses(e) = Losses(e) + log(Y(i));
        else
            Losses(e) = Losses(e) + log(1-Y(i));
        end
    end
   Lf(e)= -Losses(e)/n;

    % calcul de la fonction de perteobjectif

    % calcul du gradient de cette fonction objectifs 
    dw = zeros(1,2);
    for i = 1:n
        dw = dw + ( Y(i) - C(i) ) * (X(:,i)' - w);
    end
    dw = -2*dw/n;

    db = 0;
    for i = 1:n
        db = db + (Y(i) - C(i)) * b;
    end
    db = -2*db/n;

    % Faire la mise � jour: 
   
    w = w - lr * dw; 
    b = b - lr * db;
    
end
figure(3);
plotdata(X,C,b,w);
title('Optimized ellipse');

%Taux d'erreur de classification
figure(4);
plot(taux_err/n/100);
% title('Taux d''erreur au cours de la boucle');
ylabel('Error rate in %');
xlabel('Number of Iteration');

%Fonction Losses
figure(5);
plot(Lf);
% title('Fonction losses au cours de la boucle');
ylabel('Loss Fonction');
xlabel('Number of Iteration')
%Norme de w0+w
figure(6);
plot(norme);
xlabel('Number of iteration')
ylabel('Norm of w_0 and w');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fonction de repr閟entation graphique des donn閑s 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotw(bias, w)
fplot(@(x) -(bias + w(1)*x )/w(2), [0 20])
end

function plotdata(MX, MY,b,w)
    neg = MY==0;
    pos = MY==1;
    plot(MX(1,neg), MX(2,neg), 'r.'); hold on;  
    plot(MX(1,pos), MX(2,pos), 'g+');
    
    XCentre = w(1);
    YCentre = w(2);
    Rayon = b;
    VTheta = 0:1:360;
    VTheta = VTheta*pi/180;
    XCercle = XCentre + Rayon * cos(VTheta);
    YCercle = YCentre + Rayon * sin(VTheta);
    plot(XCercle, YCercle)
    xlim([0 20])
    ylim([0 20])
    hold off
end

function sigma = sigmoid(a)
    sigma = 1./(1+exp(-a));
end