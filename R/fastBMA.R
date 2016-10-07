fastBMA <- function( data,
					 nTimePoints = 1,
                     prior.prob = NULL,
                     genenames=colnames(data),
                     verbose = FALSE,
                     control = fastBMAcontrol()){
  #print("fastBMA inner algorithm:")
  #print(genenames)
  cl <- match.call();
  
  data <- as.matrix(data)
  
  if (is.null(control$fastgCtrl$g0)) {g0=ncol(data)}
  else {g0 <- control$fastgCtrl$g0}
  
  
  ## For verbose mode, so don't have to check every time
  vprint <- function(...){}
  if ( verbose ) {
    vprint <- function(...){print(...)}
  }
  
  # null s or 1 prior.probs
  if ( is.null(prior.prob) ) {
    prior.prob <- matrix(2.76/6000.0, ncol(data), ncol(data));
  }
  if ( length(prior.prob) == 1 ) {
    prior.prob <- matrix(prior.prob, ncol(data), ncol(data));
  }
  eps <- 1e-8;
  prior.prob = pmin( pmax( prior.prob, eps ), 1 - eps );

  #null genenames
  if (is.null(genenames)) {
    genenames <- paste("X", 1:ncol(data), sep = "")
  }
  
  #print("gene Names: ")
  #print(genenames);
  
  nvar <- ncol(data);

  #cpp call fast BMA
  vprint( "Running fastBMA..." );
  #print( "Running fastBMA..." );
  
  optimizeBit <- control$fastgCtrl$optimize

  #print(dim(data))
  #print(dim(prior.probs))
  #print(length(control$OR))
  #print(length(g))
  #print(length(genenames))
  #print(length(control$nThreads))
  #print(length(optimizeBit))
  #print(length(control$gCtrl$iterlim))
  #print(length(control$timeSeries))
  #print(length(control$noPrune))
  #print(length(control$edgeMin))
  #print(length(control$edgeTol))
  #print("start C++ fastBMA...")
  rcpp.results <- fastBMA_g(data, nTimePoints, prior.prob,
                            control$OR, control$nVars,g0,
                            genenames, control$start,
							control$end, control$nThreads, optimizeBit,
                            control$fastgCtrl$iterlim,
                            control$timeSeries, control$rankOnly,
							control$noPrune, control$edgeMin,
							control$edgeTol, control$pruneFilterMin,
							control$timeout)
  
  #print("done C++ fastBMA...");
  #print(rcpp.results)
  vprint( "Done" );
  

	edgeWeights <- rcpp.results$edgeWeights
	if (control$noPrune) edgeWeights <- abs(edgeWeights)
	nParents <- rcpp.results$nParents
	parents <- rcpp.results$parents+1

	#print("edgeWeights")
	#print(edgeWeights)
	#print("nParents")
	#print(nParents)
	#print("parents")
	#print(parents)

	edgeList <- list()
	indexCounter <- 1

	#print("edgeWeights")
	#print(edgeWeights)
	#print("nParents")
	#print(nParents)
	#print("parents")
	#print(parents)

	#print("nParents length:")
	#print(length(nParents))
	for (i in 1:length(nParents)) {
	  if(nParents[i] != 0) {
		tmpWeight <- edgeWeights[indexCounter:(indexCounter+nParents[i]-1)]
		tmpParentsInd <- parents[indexCounter:(indexCounter+nParents[i]-1)]
		tmpParents <- genenames[tmpParentsInd]
		#print("tmpWeight")
		#print(tmpWeight)
		#print("tmpParentsInd")
		#print(tmpParentsInd)
		
		if (control$showPrune) {
			selectEdgeIndex <- (abs(tmpWeight) >= control$edgeMin) & (control$selfie | tmpParentsInd != i)
		} else {
			selectEdgeIndex <- (tmpWeight >= control$edgeMin) & (control$selfie | tmpParentsInd != i)
		}
		
		#print("selectEdgeIndex")
		#print(selectEdgeIndex)
		tmpWeight <- tmpWeight[selectEdgeIndex]
		tmpParents <- tmpParents[selectEdgeIndex]
		
		names(tmpWeight) <- tmpParents
		indexCounter <- indexCounter + nParents[i];
		edgeList[[i]] <- tmpWeight
	  } else {
		edgeList[[i]] <- NULL
	  }
	}
	
	geneLength = length(edgeList)
	if (geneLength > 0)
	  names(edgeList) <- genenames[1:geneLength]

	return (edgeList)

  
}

