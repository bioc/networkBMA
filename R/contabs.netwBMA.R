contabs.netwBMA <-
function (network, reference, known=NULL, thresholds = NULL)
{

   prelim <- contabs.prelim( network, reference, known)

   if (is.null(thresholds)) {
     prob <- prelim$network[,3]
#    thresholds <- sort(unique(prob))
     thresholds <- sort(as.numeric(names(table(prob))))
   }

   if (any(thresholds > 1)) stop("thresholds: use probabilities rather than percentages")

   contabs( network = prelim$network, reference = prelim$reference, 
            size = prelim$size)
}
