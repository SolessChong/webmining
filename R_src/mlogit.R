panel.datasm = data.frame(
    cbind( 
        sample(1991:1993, 100, replace=TRUE), 
        rep(1:20,each=5), runif(100, min=0, max=1), 
        runif(100, min=0, max=6), 
        runif(100, min=2, max=6) , 
        runif(100, min=0, max=1), 
        runif(100, min=0, max=6), 
        runif(100, min=2, max=6)  ))
names(panel.datasm) = c("choice", "id", "data_1991","data_1992",
    "data_1993", "data2_1991", "data2_1992","data2_1993") 

#The mandatory arguments are choice, which is the variable that indicates the choice
#made, the shape of the original data.frame and, if there are some alternative specific
#variables, varying which is a numeric vector that indicates which columns contains
#alternative specific variables. This argument is then passed to reshape that coerced the
#original data.frame in \long" format. Further arguments may be passed to reshape. For
#example, if the names of the variables are of the form var:alt, one can add sep = ’:’.
logit.data <- mlogit.data(panel.datasm, id = "id", choice = "choice", 
    varying= 3:5, shape = "wide", sep = "_") 

head(logit.data)


#another example of mlogit
data("Fishing", package = "mlogit")
head(Fishing, 3) #wide format
## There are four fishing modes (beach, pier, boat, charter), 
## two alternative specific variables (price and catch) 
## and one choice/individual specific variable (income)
Fish <- mlogit.data(Fishing, shape="wide", varying=2:9, choice="mode") #change into long format
																		#choice" variable is now a logical variable and the individual specific variable #(income) is repeated 4 times
head(index(Fish))														#chid-->choice index, alt-->alternatives index



##Score test
library("mlogit")
data("TravelMode", package = "AER")
ml <- mlogit(choice ~ wait + travel + vcost, TravelMode,
             shape = "long", chid.var = "individual", alt.var = "mode")
hl <- mlogit(choice ~ wait + travel + vcost, TravelMode,
             shape = "long", chid.var = "individual", alt.var = "mode",
             method = "bfgs", heterosc = TRUE)
lrtest(ml, hl)
waldtest(hl)
scoretest(ml, heterosc = TRUE)
