# Libralato_data
Analis of Libralato et al data
## on branch: main

1. well_measured.py. Select well measured stars following the criteria in Libralato et al. 2021. (Except for the criteria f, that I dont know how to implement). *Have to be run for epoch 1 and 2 separatly*
2. well_measured_PM.py. Apply a couple more of condition from Libralto et al for PM selection. It returns a list with the selected stars: their proper motion and their magnitude mF139.
___
3. After step 2 you have to go to TopCat and cross match the NSD central region catalog, with the list retrived for step 2. (IN format CVS-non header)
___
4. error_Mc.py. Transform coordinates to galactic and apply a MC method for uncertainties  
