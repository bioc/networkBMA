contabs.prelim <-
function (network, reference, known=NULL)
{
 
     geneNames <- function(df2) {
      as.vector(apply(df2[, 1:2], 2, as.character))
     }

     pasteEdges <- function(df2, index=NULL) {

        if (is.null(index)) {
          nam <- sort(unique(c(as.character(df2[,1]),as.character(df2[,2]))))
#         index <- 1:length(nam)
          index <- nam
          names(index) <- nam
        }

        as.vector(apply(cbind(index[as.character(df2[, 1])],
                              index[as.character(df2[, 2])]),
                        1, paste, collapse = "#"))
    }

    if (is.null(dim(network))) stop("network should be a data frame")
    if (is.null(dim(reference))) stop("network should be a data frame")
    if (ncol(reference) > 2) stop("reference should have only two columns")

    network <- as.data.frame(network)
    reference <- as.data.frame(reference)

    keep <- !duplicated(network)
    network <- network[keep,,drop=FALSE]     

    if (ncol(network) > 2) {
        prob <- network[, 3] 
        network <- network[prob != 0,]
        prob <- prob[prob != 0]
        network <- network[,1:2]
    }
    else {
      prob <- rep(1, nrow(network))
   }

   if (any(prob > 1)) stop("network: use probabilities not percentages")

   netRG <- lapply( network[,1:2], function(x) unique(as.character(x)))

   keep <- !duplicated(reference)
   reference <- reference[keep,,drop=FALSE]     
   refRG <- lapply( reference[,1:2], function(x) unique(as.character(x)))

   nam <- sort(unique(c(unlist(netRG),unlist(refRG),
               if (!is.null(known)) geneNames(known) else NULL)))
   index <- 1:length(nam)
   names(index) <- nam

   if (!is.null(known)) {
# check that known edges are in network and reference

     keep <- !duplicated(known)
     known <- known[keep,1:2,drop=FALSE]

     knownRG <- apply( known[,1:2], 2, function(x) unique(as.character(x)))

     knownEdges <- pasteEdges(known,index)
     netEdges <- pasteEdges(network,index)
     refEdges <- pasteEdges(reference,index)

     if (length(setdiff(knownEdges,netEdges)) != 0) stop("some known edges are not in network")

# remove known edges from consideration 

     names(prob) <- netEdges
     k <- !as.logical(match( netEdges, knownEdges, nomatch = 0))
     network <- network[k,,drop=FALSE]
     netEdges <- netEdges[k]
     prob <- prob[netEdges]

     netRG <- lapply( network[,1:2], function(x) unique(as.character(x)))

     k <- !as.logical(match( refEdges, knownEdges, nomatch = 0))
     refEdges <- refEdges[k]
     reference <- reference[k,,drop=FALSE]
     refRG <- lapply( reference[,1:2], function(x) unique(as.character(x)))
   }

   names(refRG) <- names(netRG) <- c("reg","gen")

          {
                       reg <- intersect(netRG[[1]],refRG[[1]])
                       gen <- intersect(netRG[[2]],refRG[[2]])
                       if (length(reg) == 0  || length(gen) == 0) stop("network and reference dont intersect")

                       r <- as.logical(match( network[,1], reg, nomatch = 0)) 
                       g <- as.logical(match( network[,2], gen, nomatch = 0))  
                       if (!any(r & g)) {
                          warning("network and reference dont intersect")
                          return(NULL)
                       }
                       network <- network[r & g, ,drop=FALSE]
                       prob <- prob[r & g]

                       r <- as.logical(match( reference[,1], reg, nomatch = 0))
                       g <- as.logical(match( reference[,2], gen, nomatch = 0))
                       if (!any(r & g)) stop("network and reference dont intersect")
###                    reference <- reference[r & g, ,drop = FALSE]
                     }

   size <- length(refRG$reg)*length(refRG$gen)

   names(prob) <- NULL
   netsubset <- cbind(network, prob)
   refsubset <- reference

   if (!is.null(known)) {

# adjust size for any known edges that have been counted as non edges

     k <- apply( known, 1, function(x) as.numeric(any(reg == x[1]) && any(gen == x[2])))

     size <- size - sum(k)

   }

   list( network = netsubset, reference = refsubset, size = size)
}
