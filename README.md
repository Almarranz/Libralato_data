# Libralato_data
Analisis of Libralato et al data
## on branch: main
### On branch 'main' you can select whatever you want to trimm the data in same fashion Libralato et all do, or not. There is no need for the relexed_selection branch anymore.
_
## Trimming of the data: from 0 to 4 are common steos for all different aproach, since this is just the selection of the data.

__
0. error_Mc.py. Transform proper motions to galactic and apply a MC method for uncertainties . This script is run on the imac becouse it takes forever. 
> **WARNING** error_MC.py: use only the version in the iMac.  
___

0. error_gal.py. Transform proper motions to galactic and transforms the uncertainties in the same way. This should be enought as an aproximation.
1. galactic_pm_catalog.py. Adds the galactic proper motions computed in error_Mc.py to the Libralatos´s proper motion catalog. NOt intirely sure why I did a specific script just for this.
2. well_measured.py. Select well measured stars following the criteria in Libralato et al. 2021, for the **photometric** catalogs (Except for the criteria f, that I dont know how to implement). Loops through epoch1 and 2
> note: you can choose whatever or not you what to trimm your data. If you dont trimm it at all, the list generated by this scripts will have the word *relaxed* in front in the name
3. well_measured_PM.py. Apply a couple more of condition from Libralto et al for PM selection. It returns a list with the selected stars: their proper motion and their magnitude mF139.
> Note: you can select the trimmed or the untrimmed data  
> Note: you can choose whatever or not you what to trimm your data further. If you dont trimm it at all, the list generated by this scripts will have the word *relaxed* in front in the name
4. match_w_GNS.py. Crosses match GNS central region with the lits from Libralato
> Note: here you can select the limit to cosider two stars as the same. **Check the relevance of this choice**


# Aprach A: it consist in looking around MS in the catalog_D from Libralato data. NOT very suscessful aproach so far...
___
5. co_mouving_group.py Generates lists and plots of the stars aroun the Ms stars found in the refined data. 
>NOte: in this scrip you can select stars **under a max pm uncertainty*. For both relaxed and trimmed.  

6. (A) dbscan_compartion.py. Runs DBSCAN on the lists generated by #5 script. Uses the lists generated with different radios and plot them in a way that they are easy to compare.
> Uses the minimun k-NN distance to define a first epsilon, and runs DBSCAN, ir no clusters are found at that distance, increases epsilon and run it again.  
> Alternativily uses the first codo found in the distance vs point plots and increases in loop the number of minimun points in a cluster till only a limmited number of clusters remain
6. (B) hdbscan_compartion.py. Runs Hdbscan on the lists generated by #5 script. Uses the lists generated with different radios and plot them in a way that they are easy to compare.
>Hdbscan with **'leaf'** configuration (instead of 'eom' configuration) yields to more *realistic* clusters
6. (C) dbscan_comparation_SPISEA. Same as dbscan_comparation, but plots the CMD of each cluster on the CMD of a symulated cluster with the same AKs of that of the mean AKs of the members of the cluster.

##SECOND SECUENCE
Afetr running scripts 0 to 6 (A and B) follow this secuece:


1. cluster_Aks_gns.py. Looks for each member of the putative cluster in the extintion catalog adn extract the exctintion, if the star if consider to be in the galactic centre. Produces a list with mean AKs and std'

2. AKs_for_cluster.py. Constructs a cluster using as AKs the mean AKs of the member of the selected cluster. Then we over plot on the CMD plot of the sythetic cluster, the CMD of the members of the found cluster.
> Note: ispect and select by eye the cluster than fit the best with the symulated data

3. selected_group_analysis.py . Looks around a selected cluster from dbscan_comparation.py or hdbscan_comparation.py. Then extract the parametres of the new cluster and stores it in a new list
> Note: So far yoy have to manually change the parametres of dbsdcan. 

4. selec_cluster_AKs_gns.py. Extracts (from Paco´s extintiction catalog) the AKs value for each member of selected  clusters obtained with dbscan_coparation.py or hdbscan_comparation.py, that we  will use to construct a cluster with SPISEA
>Note: basiscally same than AKs_for_cluster.py, but for the clusters than results of runnib dbscan with one particular cluster in the center and not one Ms

5. select_AKs_for_cluster.py. Same of AKs_for cluster.py but for an extended cluster 
> Note: At this point this in when I decide if I have a cluster or not.
___

# Aproach B: it consists in dividing the data in 4 bigger section and diviede each section in smaller section and look for the clusters. IN PROGRESS
___
5. sections_A_to_D.py. Divide the data in the most obvious 4 sections and stores them in four lists. This are the star the matched the GNS (from match_w_GNS.py.), not yet trimmed in velocity or color
6. dbs_kernel_subsecA.py.(A to D) divides each section in subsections. Runs dbscan with the kernel method and stores common clusters in common folders'
7. BAN_dbs_kernel_subsecA.py.(A to D). DO the same thing. 


## Other scripts

Arches_Hosek.py. In this iscript arches (or quintuplet) are dbscaned for clustering. 
>In a first step the whole data set is scanned and withdraw the denser cluster.
>Afterwards a reduced data set is scanned for comparations. 
>Selecting a mini cluster at the center of the found cluster we can have an idea of how a almost disolving cluster might look like in the galactic center.
>Selecting at the center improves the posibilities of choosing  mini cluster without any false cluster member in it.
>In this mini cluster we can look for the dispersion in velocity and color to have an idean of what we are looking for in Lkbralato data.

Arches_Hosek_Kernel.py 
>Finds the core of the Arches or Quintuplet cluster from Hosek data using the kernel method and dbscan. Then matches it with GNS and fit an isochrone and a simulated cluter to it.
Arches_Hosek_Kernel.py 
>Finds cluster with dbsacn usin the kernel simulation for the epsilon. Then selecte a small group of stars from the  middle of the cluster and plots then. This is for comparation of how a cluster in the NSD with only a few stars should look like
same_cluster_diff_MS.py. 
>Fins same cluster around different MS **whithin the same search(dbscan or hdbscan)**. 


cluster_to_region.py. 
>Generates a region of the selected cluster to be visualized in DS9.

candidates_plotting.py (and BAN_candidates_plotting.py). 
>Put toghter and plots all clusters found by dbs_kernel_subsecA.py (or BAN_dbs_kernel_subsecA) adn plot them
Lib_vs_Ban_comp.py. 
>Uses the folder geneated by previous step (or any other folder with files containing cluster information for that matter) for plotting together (in 5D space) cluster closer that certain separation 







