contabs.netwBMA <-
function (network, reference, known=NULL, threshholds = NULL)
{

   prelim <- contabs.prelim( network, reference, known)

   if (is.null(threshholds)) {
     prob <- prelim$network[,3]
#    threshholds <- sort(unique(prob))
     threshholds <- sort(as.numeric(names(table(prob))))
   }

   if (any(threshholds > 1)) stop("threshholds: use probabilities rather than percentages")

   contabs( network = prelim$network, reference = prelim$reference, 
            size = prelim$size)
}
