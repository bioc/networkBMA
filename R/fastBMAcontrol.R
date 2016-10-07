fastBMAcontrol <- function(OR = 10000,
                           timeSeries = TRUE,
						   rankOnly = FALSE,
                           noPrune = FALSE,
                           edgeMin = 0.5,
                           edgeTol = -1,
                           nThreads = 1,
						   selfie = FALSE,
						   showPrune = FALSE,
						   nVars = 0,
						   fastgCtrl = fastgControl(),
						   start = -1,
						   end = -1,
						   pruneFilterMin = 0,
						   timeout = 0) {
  
  list( algorithm = "fastBMA", OR = OR, timeSeries=timeSeries, rankOnly = rankOnly,
		noPrune = noPrune, edgeMin = edgeMin, edgeTol = edgeTol,
		nThreads = nThreads, selfie = selfie, showPrune = showPrune,
		nVars = nVars, fastgCtrl = fastgCtrl, start = start, end = end,
		pruneFilterMin = pruneFilterMin, timeout = timeout);
  
  
  
}

fastgControl <- function(optimize = 0, g0 = NULL, iterlim = 20) {
	list(optimize = optimize, g0 = g0, iterlim = iterlim)
}