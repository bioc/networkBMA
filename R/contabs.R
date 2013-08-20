contabs <-
function (network, reference, size, thresholds = NULL) 
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

    list2df <- function(edgeList, full = FALSE) {
        onam <- names(edgeList)
        if (is.numeric(edgeList[!sapply(edgeList, is.null)][[1]])) {
            prob <- as.vector(unlist(edgeList))
            edgeList <- lapply(edgeList, names)
        }
        else prob <- NULL
        inam <- unlist(edgeList)
        edgeList <- lapply(edgeList, unlist)
        lev <- union(onam, unique(inam))
        if (full) {
            ll <- length(lev)
            mat <- data.frame(regulator = as.factor(rep(lev, 
                ll)), gene = as.factor(rep(lev, each = ll)))
            if (!is.null(prob)) {
                stop("not fixed to handle probabilities")
            }
        }
        else {
            if (is.null(prob)) {
                mat <- data.frame(regulator = factor(inam, levels = lev), 
                  gene = factor(rep(onam, sapply(edgeList, length)), 
                    levels = lev))
            }
            else {
                mat <- data.frame(regulator = factor(inam, levels = lev), 
                  gene = factor(rep(onam, sapply(edgeList, length)), 
                    levels = lev), prob = prob)
            }
        }
        rownames(mat) <- NULL
        mat
    }

    if (is.null(dim(network))) 
        network <- list2df(network)

    if (ncol(network) == 3) {
        prob <- network[, 3] 
        network <- network[prob != 0,]
        prob <- prob[prob != 0]
    }
    else {
      prob <- rep(1, nrow(network))
   }

   if (any(prob > 1)) stop("network: use probabilities not percentages")

   if (is.null(dim(reference))) reference <- list2df(reference)

   if (ncol(reference) > 2) stop("reference should have only two columns")

   net.reg <- unique( as.character(network[,1]))
   net.gen <- unique( as.character(network[,2]))

   ref.reg <- unique( as.character(reference[,1]))
   ref.gen <- unique( as.character(reference[,2]))

   nam <- sort(unique(c(geneNames(network),geneNames(reference))))
   index <- 1:length(nam)
   names(index) <- nam

   network <- pasteEdges( network, index)

   reference <- pasteEdges( reference, index)

   if (is.null(thresholds)) {
#      thresholds <- sort(unique(prob))
     thresholds <- sort(as.numeric(names(table(prob))))
   }

   if (any(thresholds > 1)) stop("thresholds: use probabilities rather than percentages")

   thresholds <- sort(thresholds)

   TP <- FN <- FP <- TN <- rep(NA, length(thresholds))

#  labls <- as.character( sapply( as.character(thresholds), 
#                           function(x) substring( x, 1, min(nchar(x),8))))

   labls <- as.character(thresholds)
   names(TP) <- names(FN) <- names(FP) <- names(TN) <- labls

   nMissing <- size - length(reference)
   for (i in seq(along = thresholds))
      {
        thresh <- thresholds[i]
        index <- labls[i]
        if (thresh != 0) {
          edges <- network[prob >= thresh]
          TP[index] <- length(intersect(edges, reference))
          FP[index] <- length(setdiff(edges, reference))
          FN[index] <- length(setdiff(reference, edges))
          TN[index] <- nMissing - FP[index]
        }
        else {
# all pairs are edges
          TP[index] <- length(reference)
          FP[index] <- size - TP[index]
          FN[index] <- 0
          TN[index] <- 0
        }
        if (any(c(TP[index],FP[index],FN[index],TN[index]) < 0)) {
            print(c(TP = TP[index], FP = FP[index], FN = FN[index], 
                TN = TN[index]))
            stop("size must be incorrect")
        }
   }

   data.frame(TP = TP, FN = FN, FP = FP, TN = TN, row.names = labls)
    
}
