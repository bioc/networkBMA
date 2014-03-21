## Edited to allow tie breakers (aks)
list2df <- function(edgeList, aks = NULL, full = FALSE) {
  onam <- names(edgeList)
  tiebreak <- NULL;
  if (!is.null(aks) && is.numeric(edgeList[!sapply(aks, is.null)][[1]])) {
    tiebreak <- as.vector(unlist(aks))
  }
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
      colnames(mat) <- c( "Regulator", "TargetGene" );
    }
    else {
      if ( is.null(tiebreak) ) {
        mat <- data.frame(regulator = factor(inam, levels = lev), 
                          gene = factor(rep(onam, sapply(edgeList, length)), 
                          levels = lev), post.prob = prob);
        colnames(mat) <- c( "Regulator", "TargetGene", "PostProb" );
      }
      else {
        mat <- data.frame(regulator = factor(inam, levels = lev), 
                          gene = factor(rep(onam, sapply(edgeList, length)), 
                          levels = lev), post.prob = prob, tiebreak = tiebreak);
        colnames(mat) <- c( "Regulator", "TargetGene", "PostProb", "Tiebreak" );
      }
    }
  }
  rownames(mat) <- NULL
  mat
}
