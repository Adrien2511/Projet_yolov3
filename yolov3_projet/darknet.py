from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def lecture_cfg(cfgfichier):#lecture et division des blocks du fichier cfg


    file = open(cfgfichier, 'r')#ouverture du fichier
    lignes = file.read().split('\n')
    lignes = [x for x in lignes if len(x) > 0]#supression des lignes vides
    lignes = [x for x in lignes if x[0] != '#']#supression des commentaires
    lignes = [x.rstrip().lstrip() for x in lignes] #supression des espaces à gauche et à droite

    # division de la liste en différent groupe

    groupe = {}
    groupes = []


    for ligne in lignes:
        if ligne[0] == "[":  #chaque groupe commence par un crochet
            if len(groupe) != 0:  # If block is not empty, implies it is storing values of previous block.
                groupes.append(groupe)
                groupe = {}  #on remet à 0 le dico
            groupe["type"] = ligne[1:-1].rstrip() #on prends tout les éléments sauf le premier et le dernier pour enlever les crochets et on fixe la valeur du type
        else:
            key, value = ligne.split("=")
            groupe[key.rstrip()] = value.lstrip() #on donne une valeur à chaque clé une valeur
    groupes.append(groupe)                        #on ajoute le dictio au vecteur

    return groupes

groupes = lecture_cfg("cfg/yolov3.cfg")
print(groupes)

def creation_des_modules(groupes):
    info_net = groupes[0]
    list_module = nn.ModuleList() #nn.Module est une classe pour les réseaux de neurone et ici nous avons les informations dans une list
    filtre_p = 3 # l'image est en coueleur (RVB)
    filtre_sortie=[] #permet de retenir les filtres précédent

    for num,y in enumerate(groupes[1:]):
        mod = nn.Sequential()
        if (y["type"]=="convolutional"): #on regarde si c'est un type convolutif
            activation = y["activation"]
            try:
                batch_normalize = int(y["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            pad = int(y["pad"])  #savoir si on a ou pas du padding
            size_Kernel = int(y["size"]) #taille des noyaux
            stride = int(y["stride"]) #pour savoir le déplacment du noyau à travers le canal d'entrée (x,y)
            filters = int(y["filters"]) # le nombre de noyaux

            if pad:             #fix le padding si il y en a un
                pad = (size_Kernel-1)//2 #calcul pour trouver la taille du padding par rapport au noyau afin d'avoir une sortie de la taille de l'entrée
            else :
                pad = 0
            conv = nn.Conv2d(filtre_p, filters, size_Kernel, stride, pad, bias=bias) #création du réseaux de convolution
            mod.add_module("conv_{0}".format(num), conv) #ajoute le module 
















