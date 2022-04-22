# Libralato_data
Analisis of Libralato et al data
## on branch: main
### Branch main follows the paper of Libralato et al. 2021 for the selection of the good data, and uses the scripts erro_Mc for the transformatipm of the galactic coordenates and errors afterwards
0. error_Mc.py. Transform coordinates to galactic and apply a MC method for uncertainties . This script is run on the imac becouse it takes forever. 
1. galactic_pm_catalog.py. Adds the galactic proper motions computed in error_Mc.py to the Libralatos´s proper motion catalog. NOt intirely sure why I did a specific script just for this.
2. well_measured.py. Select well measured stars following the criteria in Libralato et al. 2021, for the **photometric** catalogs (Except for the criteria f, that I dont know how to implement). *Have to be run for epoch 1 and 2 separatly*
> note: you can choose whatever or not you what to trimmed your data. If you dont trimmit at all, the list generated by this scripts will have the word *relaxed* in front in the name
3. well_measured_PM.py. Apply a couple more of condition from Libralto et al for PM selection. It returns a list with the selected stars: their proper motion and their magnitude mF139.
> Note: you can select the trimmed or the untrimmed data

> Note: you can choose whatever or not you what to trimm your data further. If you dont trimm it at all, the list generated by this scripts will have the word *relaxed* in front in the name
4. match_w_GNS.py. Crosses match GNS central region with the lits from Libralato


5. co_mouving_group.py Generates lists and plots of the stars aroun the Ms stars found in the refined data. 
6. dbscan_compartion.py. Runs DBSCAN on the lists generated by #5 script. Uses the lists generated with different radios and plot them in a way that they are easy to compare.
> Uses the minimun k-NN distance to define a first epsilon, and runs DBSCAN, ir no clusters are found at that distance, increases epsilon and run it again.

> Alternativily used the frist codo found in the distance vs point plots and increases in loop the number of minimun points in a cluster till only a limmited number of clusters remain

## Other scripts

Arches_Hosek.py. In this iscript arches (or quintuplet) are dbscaned for clustering. 
>In a first step the whole data set is scanned and withdraw the denser cluster.
>Afterwards a reduced data set is scanned for comparations. 
>Selecting a mini cluster at the center of the found cluster we can have an idea of how a almost disolving cluster might look like in the galactic center.
>Selecting at the center improves the posibilities of choosing  mini cluster without any false cluster member in it.
>In this mini cluster we can look for de dispersion in velocity and color to have an idean of what we are looking for in Lkbralato data.