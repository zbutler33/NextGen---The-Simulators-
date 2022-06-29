# setwd("C:/Users/k322e071/OneDrive - The University of Kansas/CUAHSI/R_Job")
# getwd()

#***************** Install Packages ****************#
install.packages("dataRetrieval")
library(dataRetrieval)
vignette("dataRetrieval", package = "dataRetrieval")

# Load/call data by states (Discharge[ft3/s])
# Retrieve all Gages in Maine
Mn_sites <- whatNWISsites(stateCd = "ME", 
                          parameterCd = "00060")
# Display dataframe
names(Mn_sites)

Mn_df <- data.frame(Mn_sites)
print(Mn_df)

# Export all intersected Gages as csv
Mn_Gages <- "C:\\Users\\k322e071\\OneDrive - The University of Kansas\\CUAHSI\\R_Job\\Maine_USGS.csv"
write.csv(Mn_df,Mn_Gages, row.names = FALSE)
