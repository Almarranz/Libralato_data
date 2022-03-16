# Libralato_data
Analisis of Libralato et al data
___
## on branch: relaxed selection
___
### Branch *relaxed_selection* uses erro_Mc on the whole catalog for Libralato as first steep, the works on the trimming of the data, but in a different way that Libralato does 
1. error_Mc.py. Transform coordinates to galactic and apply a MC method for uncertainties . This script is run on the imac becouse it takes forever. 

2. well_measured.py. Here you can choose between **trim or not** the data in Libralato catalogs. 
3. well_measured_PM.py. Apply a couple more of condition from Libralto et al for PM selection. It returns a list with the selected stars: their proper motion and their magnitude mF139.
___
4. After step 3 you have to go to **TopCat**, look in **VIZIER** for the NSD central region catalog (DO NOT forget to check **unlimited rows**), and cross-match it  with the list retrived for step 3. (IN format CVS-non header)
___
5. Now we can do the fitting of the Gaussian to the data for mul and mub. This steep *is not necessary* for steep 6
> - 5.1. vy_gsa_fit.py. Ypu have to choose the color cut and the max allowed uncertaintie in velocity.
> - 5.2. vy_gsan_fit.py. Same as 5.1 but for the parallel componet. This one uses 3 Gaussian. Also you can use vx_2gas_fit.py, that uses 2 Gaussian for the fitting as the did in Libralato et al. 2021.
6. co_mouving_group.py Generates lists and plots of the stars aroun the Ms stars found in the refined data. 
7. DBSCAN_tests.py. Runs DBSCAN on the lists generated by #5 script.
