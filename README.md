# Libralato_data
Analisis of Libralato et al data
## on branch: relaxed selection
### Branch *relaxed_selection* uses erro_Mc on the whole catalog for Libralato as first steep, the works on the trimming of the data, but in a different way that Libralato does 
1. error_Mc.py. Transform coordinates to galactic and apply a MC method for uncertainties . This script is run on the imac becouse it takes forever. 

2. well_measured.py. Select well measured stars following the criteria in Libralato et al. 2021, for the **photometric** catalogs (Except for the criteria f, that I dont know how to implement). *Have to be run for epoch 1 and 2 separatly*
2. well_measured_PM.py. Apply a couple more of condition from Libralto et al for PM selection. It returns a list with the selected stars: their proper motion and their magnitude mF139.
___
3. After step 2 you have to go to TopCat and cross match the NSD central region catalog, with the list retrived for step 2. (IN format CVS-non header)
___
5. co_mouving_group.py Generates lists and plots of the stars aroun the Ms stars found in the refined data. 
6. DBSCAN_tests.py. Runs DBSCAN on the lists generated by #5 script.
