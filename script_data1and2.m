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
clc;
clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chargement des donn閑s 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load reglog_data_1.mat;
% Vous disposez d閟ormais de X et C 
% Regarder les donn閑s : contenu, dimensions, ... 
% On peut aussi les repr閟enter sur une figure:
n=length(X)
figure(1)
plotdata(X,C);
legend('C=0','C=1')

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
figure(2) % pour obtenir une nouvelle figure
plotdata(X,C,w0,w);
hold on 

% inf閞ence : calculer les probabilit閟 d'appartenir � la classe 1,
Y = zeros(1,200);
Y = 1./(1+exp(-w0-w*X));

% selon le mod鑜e de param鑤res w0 et w, pour chaque exemple de X
L = -1/length(Y)*sum(C.*log(Y)+(1-C).*log(1-Y));

% calcul de la fonction de perteobjectif
% calcul du gradient de cette fonction objectifs 
dw = -1/length(Y)*sum((C-Y).*X,2);
db = -1/length(Y)*sum(C-Y,2);
% % Faire la mise � jour: 
w = w - lr * dw'; 
w0 =  w0-lr*db; 
% Repr閟enter la nouvelle droite et jouer avec le pas d'apprentissage 
% pour obtenir une nouvelle figure
plotdata(X,C,w0,w);
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Boucle d'apprentissage et monitoring: 
% Nous pouvons maintenant mettre en place la boucle d'apprentissage
% Le nombre d'閜oque d'apprentissage est fix閑 par une variable. 
% L'objectif est d'observer l'関olution de certaines grandeurs 
% au cours de l'apprentissage, en particulier 
% l'関olution de la fonction de perte que l'on cherche � minimiser. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nepochs = 10000;

Losses = zeros(1,Nepochs);
errors = zeros(1,Nepochs);

for e=1:1:Nepochs

    Y = 1./(1+exp(-w0-w*X)); %calculate Y by sigmoid function
            
    errors(e)= sum(abs((Y>0.5)-C));    % calculte error rate
        
    L = -1/length(Y)*sum(C.*log(Y)+(1-C).*log(1-Y)); %calculation of loss function
    Losses(e) = L ;
    dw = (-1/length(Y))*sum((C-Y).*X,2);  %after calculation mannually we got the expression of dw and db
    db = (-1/length(Y))*sum(C-Y,2); %expression for delta w_0
    
    w =  w - lr * dw';  % apply the change by gradient decent method
    w0 = w0 - lr*db;  
    norme(e) = sqrt(w0^2+w*w'); % calcultae its norm
end
%figure of logistic regression
figure(3)
plotdata(X,C,w0,w);
legend('C=0','C=1','Linear Regression')
%figure of loss fuction
figure(4)
plot(Losses);
ylabel('Loss Function');
xlabel('Number of Iteration')
%figure of error rate
figure(5);
plot(errors);
xlabel('Number of Iteration')
ylabel('Error rate in %');

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
    if nargin == 4
        plotw(b,w)
    end
    xlim([0 20])
    ylim([0 20])
    hold off
end
