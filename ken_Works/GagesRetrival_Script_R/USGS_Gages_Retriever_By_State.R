# setwd("C:/Users/k322e071/OneDrive - The University of Kansas/CUAHSI/R_Job")
# getwd()

#***************** Install Packages ****************#
install.packages("dataRetrieval")
library(dataRetrieval)
vignette("dataRetrieval", package = "dataRetrieval")

# Load/call data by states (Discharge[ft3/s])
# Retrieve all Gages in Maine
ST_Gages = whatNWISsites(stateCd = "MA", 
                          parameterCd = "00060")
# Display dataframe
names(ST_Gages)

Gage_df = data.frame(ST_Gages)
print(Gage_df)

# Export all intersected Gages as csv
OutputHolder = "C:\\Users\\k322e071\\OneDrive - The University of Kansas\\CUAHSI\\R_Job\\Massachusetts.csv"
write.csv(Gage_df, OutputHolder, row.names = FALSE)
