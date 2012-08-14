networkBMA <-
function(data, nTimePoints, prior.prob = NULL, known = NULL, 
         ordering = "bic1+prior", nvar = NULL, self = TRUE, 
         maxreg = NULL, control = iBMAcontrolLM(thresProbne0 = 5),
         verbose = FALSE) {

if (!exists("bicreg")) library("BMA")

prob <- NULL
if (!is.null(prior.prob)) size <- length(prior.prob) == 1  # model size prior

data <- as.matrix(data)
mvar <- c(nrow(data)-1,ncol(data))
nvar <- if (is.null(nvar)) mvar else min(mvar,nvar)

nGenes <- ncol(data)
if (is.null(maxreg)) maxreg <- nrow(data)-1

vars <- 1:nGenes
edgeList <- rep( list(1:maxreg), length(vars))
names(edgeList) <- colnames(data)[vars]

nReplicates <- nrow(data)/nTimePoints
Times <- 1:nTimePoints
yIndex  <- which(as.logical(match( rep(Times, nReplicates), Times[-1], 
                            nomatch = 0)))

# xIndex is yIndex - 1

k <- 0
for (i in vars) {
   k <- k + 1
  
   gene <- colnames(data)[i]
   y <- as.numeric(data[yIndex,i])

# get the known regulators
   iKnown <- NULL
   if (!is.null(known)) {
     prior.reg.vec <- unique(known[which(known[,2] == gene), 1])
     iKnown <- match( prior.reg.vec, colnames(data), nomatch = 0)
     if (any(iKnown == 0)) stop("iKnown regulator not found in data")
     names(iKnown) <- prior.reg.vec
   }

   hasNA <- apply(is.na(data[yIndex-1,,drop=FALSE]), 1, any) | is.na(y)
   xIndex <- yIndex[!hasNA] - 1
   y <- y[!hasNA]

   if (length(iKnown) == 0) {

     if (!is.null(prior.prob)) {
       if (size) {
         prob <- rep(prior.prob,ncol(data))
       }
       else {
         prob <- as.numeric(prior.prob[, gene])
       }
     }

     ord <- varord( x=data[xIndex,,drop=FALSE], y=y, prior.prob=prob,
                    ordering=ordering)

     names(ord) <- colnames(data)[ord]

     if (!is.null(prior.prob)) {
       ord <- c(ord[prob[ord]==1],ord[prob[ord] != 1])
       ord <- ord[prob[ord] != 0]
     }

     if (!self) ord <- ord[names(ord) != gene]

     ord <- ord[1:min(nvar,length(ord))]
     prob <- prob[ord]

     x <- data[xIndex,ord,drop=FALSE]

     ibma.val <- iterateBMAlm( x=x, y=y, prior.prob=prob,
                               control=control, verbose=FALSE)
  } 
  else {

     if (!is.null(prior.prob)) {
       if (size) {
         prob <- rep(prior.prob,ncol(data))
       }
       else {
         prob <- as.numeric(prior.prob[, gene])
       }
       prob <- prob[-iKnown]
     }

# remove iKnown regulators from combine.genes
# Hard-coded regulators do not necessarily have high prior regulatory 
# potentials in tftp.prob, so there may be no overlap.

    knownData <-  data[xIndex,iKnown,drop=FALSE]

        residuals <- list(
            y = lm.fit(x=cbind(1,data[xIndex,iKnown,drop=F]), 
                y=as.numeric(y))$residuals,
            x = apply( data[xIndex,-iKnown,drop=F], 2,
               function(y,x) lm.fit(x=x, y=as.numeric(y))$residuals,
               x = cbind(1,data[xIndex,iKnown,drop=F])
                ))
 
    names(residuals$y) <- rownames(residuals$x) <- names(y)

    ord <- varord( x=residuals$x, y=residuals$y, 
                   prior.prob = prob,
                   ordering = ordering)

    names(ord) <- colnames(data)[-iKnown][ord]
 
    if (!is.null(prior.prob)) {
      ord <- c(ord[prob[ord]==1],ord[prob[ord] != 1])
      ord <- ord[prob[ord] != 0]
    }

    if (!self) ord <- ord[names(ord) != gene]

    ord <- ord[1:min(nvar,length(ord))]
    prob <- prob[ord]

    residuals$x <- residuals$x[,ord]

    x <- data[xIndex,colnames(residuals$x),drop=FALSE]

    ibma.val <- iterateBMAlm.resid(x=x, y=y, known = knownData, 
                                   residuals = residuals,
                                   prior.prob = prob,
                                   control = control, verbose=FALSE)

   }

  sorted.probne0 <- sort(ibma.val$probne0, decreasing=TRUE)
  nonz <- sorted.probne0 > 0
  
  edgeList[[gene]] <- if (any(nonz)) pmin(sorted.probne0[nonz]/100,1) else NULL

  if (verbose) {
    if (is.null(known)) {
      cat("gene", i, gene, "\n")
    }
    else {
      cat("gene", i, gene, "# known regulators", length(iKnown), "\n")
    }
  }

  if (verbose) print(edgeList[[gene]])
}

 result <- list2df(edgeList)
 class(result) <- c("networkBMA","data.frame")
 result
}
