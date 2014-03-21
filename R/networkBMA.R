networkBMA.old <-
function(data, nTimePoints, prior.prob = NULL, known = NULL, 
         ordering = "bic1+prior", nvar = NULL, self = TRUE, 
         maxreg = NULL, control = iBMAcontrolLM(thresProbne0 = 5),
         verbose = FALSE) {
  networkBMA( data, nTimePoints, prior.prob, known, ordering, nvar, self,
              maxreg, "iBMA", control, FALSE, FALSE, verbose );
}

networkBMA <- function(data, nTimePoints, prior.prob = NULL, known = NULL, 
                       ordering = "bic1+prior", nvar = NULL, self = TRUE, 
                       maxreg = NULL, algorithm="ScanBMA",
                       control = ScanBMAcontrol(),
                       diff0 = TRUE, diff100 = TRUE,
                       verbose = FALSE) {

## if (!exists("bicreg")) library("BMA")

  vprint <- function(...){;}
  vcat <- function(...){;}
  if ( verbose ) {
    vprint <- function(...){print(...);}
    vcat <- function(...){cat(...);}
  }
  
  ## Check provided parameters to make sure nothing is inconsistent
  ## known must be NULL for ScanBMA
  ## known must be NULL for differentiation
  if ( !is.null(known) ) {
    if ( algorithm == "ScanBMA" ) {
      stop("ScanBMA can only be used if there are no known relationships supplied.");
    }
    if ( diff0 | diff100 ) {
      stop("Differentiation can only be used if there are no known relationships supplied.");
    }
  }

  prob <- NULL
  nullPrior <- is.null(prior.prob);
  if (!nullPrior) size <- length(prior.prob) == 1  # model size prior

  data <- as.matrix(data)
  mvar <- c(nrow(data)-1,ncol(data))
  nvar <- if (is.null(nvar)) mvar else min(mvar,nvar)

  nGenes <- ncol(data)
  genenames <- colnames(data);
  if (is.null(maxreg)) maxreg <- nrow(data)-1

  vars <- 1:nGenes
  edgeList <- rep( list(1:maxreg), length(vars))
  names(edgeList) <- genenames[vars]

  ## aks are the tie-breakers for prob 100 and prob 0 edges
  aks <- rep( list(1:maxreg), nGenes);
  names(aks) <- genenames[vars];
  
  nReplicates <- nrow(data)/nTimePoints
  Times <- 1:nTimePoints
  yIndex  <- which(as.logical(match( rep(Times, nReplicates), Times[-1], 
                              nomatch = 0)))

  # xIndex is yIndex - 1

  k <- 0
  for (i in vars) {
     k <- k + 1
    
     gene <- genenames[i]
     y <- as.numeric(data[yIndex,i])

  # get the known regulators
     iKnown <- NULL
     if (!is.null(known)) {
       prior.reg.vec <- unique(known[which(known[,2] == gene), 1])
       iKnown <- match( prior.reg.vec, genenames, nomatch = 0)
       if (any(iKnown == 0)) stop("iKnown regulator not found in data")
       names(iKnown) <- prior.reg.vec
     }

     hasNA <- apply(is.na(data[yIndex-1,,drop=FALSE]), 1, any) | is.na(y)
     xIndex <- yIndex[!hasNA] - 1
     y <- y[!hasNA]

    ## refactoring - pulling out prob definition
    prob <- NULL;
    if (!nullPrior) {
      if (size) {
        prob <- rep(prior.prob,nGenes)
      }
      else {
        prob <- as.numeric(prior.prob[, gene])
      }
    }

    if (length(iKnown) == 0) {

      ord <- varord( x=data[xIndex,,drop=FALSE], y=y, prior.prob=prob,
                     ordering=ordering)

      names(ord) <- colnames(data)[ord]

      if (!is.null(prior.prob)) {
        ord <- c(ord[prob[ord]==1],ord[prob[ord] != 1])
        ord <- ord[prob[ord] != 0]
      }

      if (!self) ord <- ord[names(ord) != gene]

      fullord <- ord;
      fullprob <- prob[ord];

      ord <- ord[1:min(nvar,length(ord))]
      prob <- prob[ord]

      x <- data[xIndex,ord,drop=FALSE]

      ## run BMA
      ibma.val <- NULL;
      if ( algorithm == "ScanBMA" ) {
        ibma.val <- ScanBMA( x=x, y=y, prior.prob=prob,
                             control=control, verbose=verbose)
      }
      else {
        ibma.val <- iterateBMAlm( x=x, y=y, prior.prob=prob,
                                  control=control, verbose=verbose)
      }
      if ( diff0 | diff100 ) {
        if ( algorithm == "ScanBMA" && control$useg ) {
          fullx <- data[xIndex,fullord,drop=FALSE];
          ibma.val <- BMA.Diff.g( fullx, y, fullprob, ibma.val, diff100, diff0 );
        }
        else {
          fullx <- data[xIndex,fullord,drop=FALSE];
          ibma.val <- BMA.Diff.BIC( fullx, y, fullprob, ibma.val, diff100, diff0 );
        }
      }
    } 
    else {
      ## remove known regulators
      if (!nullPrior) {
        prob <- prob[-iKnown]
      }

      ## remove iKnown regulators from combine.genes
      ## Hard-coded regulators do not necessarily have high prior regulatory 
      ## potentials in tftp.prob, so there may be no overlap.

      knownData <-  data[xIndex,iKnown,drop=FALSE]

      ## residuals <- list(
      ##                   y = lm.fit(x=cbind(1,data[xIndex,iKnown,drop=F]), 
      ##                     y=as.numeric(y))$residuals,
      ##                     x = apply( data[xIndex,-iKnown,drop=F], 2,
      ##                     function(y,x) lm.fit(x=x, y=as.numeric(y))$residuals,
      ##                     x = cbind(1,data[xIndex,iKnown,drop=F])
      ##                     ))

      ## Generate the residuals
      residuals <- list(
                        y = lm.fit(x=cbind(1,knownData), 
                          y=as.numeric(y))$residuals,
                        x = apply( data[xIndex,-iKnown,drop=F], 2,
                          function(y,x) lm.fit(x=x, y=as.numeric(y))$residuals,
                          x = cbind(1,knownData)
                          ))
   
      names(residuals$y) <- rownames(residuals$x) <- names(y)

      ord <- varord( x=residuals$x, y=residuals$y, 
                     prior.prob = prob,
                     ordering = ordering)

      names(ord) <- genenames[-iKnown][ord]
   
      if (!nullPrior) {
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
                                     control = control, verbose=verbose)
    }

    sorted.probne0 <- sort(ibma.val$probne0, decreasing=TRUE)
    nonz <- sorted.probne0 > 0
    
    sorted.aks <- if ( diff0 | diff100 ) abs(ibma.val$ak[order(ibma.val$probne0, decreasing=TRUE)]) else NULL;

    ## Assign the found regulators to the edgelist list
    if ( any(nonz) ) {
      edgeList[[gene]] <- pmin(sorted.probne0[nonz]/100,1);
      aks[[gene]] <- if ( diff0 | diff100 ) sorted.aks[nonz] else NULL;
    }
    else {
      edgeList[[gene]] <- NULL;
      aks[[gene]] <- NULL;
    }
      
    if (is.null(known)) {
      vcat("gene", i, gene, "\n")
    }
    else {
      vcat("gene", i, gene, "# known regulators", length(iKnown), "\n") #
    }

    vprint(edgeList[[gene]]);
    vprint(aks[[gene]]);
  }

  if ( !diff0 && !diff100 ) {
    aks <- NULL;
  }

  ## Turn the edgelist list into a data frame
  result <- list2df(edgeList, aks);
  if ( nrow(result) > 0 ) {
    result <- result[order(result$PostProb, decreasing=TRUE),];
    if ( ncol(result) == 4 && sum(result$PostProb == 1) > 1 ) {
      result[result$PostProb == 1,] <- result[result$PostProb == 1,][
                                                order(result$Tiebreak[result$PostProb == 1],
                                                      decreasing = TRUE),];
    }
  }
  class(result) <- c("networkBMA","data.frame")
  result[,1:3];
}
