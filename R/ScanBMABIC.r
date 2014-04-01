ScanBMA.BIC <- function(x,
                      y,
                      prior.probs = rep(.5, ncol(x)),
                      OR = 100,
                      verbose = FALSE ) {

  ## For verbose mode, so don't have to check every time
  vprint <- function(...){;}
  if ( verbose ) {
    vprint <- function(...){print(...);}
  }
  
  delta <- 10;
  counter <- 0;
  
  n <- length(y);
  res <- list(edgelist=matrix(NA, nrow=1, ncol=3),
              modelprobs=NULL, sizes=NULL, r2s=NULL);
  bmaresult <- NULL;
  FastScanBMA.BIC( x=x, y=y, prior.probs=prior.probs, OR=OR );
}


BMA.Diff.BIC <- function( x, y, prior.probs, result, diff100, diff0 ) {
  if ( is.null(prior.probs) ) {
    prior.probs <- rep(.5, ncol(x));
  }
  if ( length(prior.probs) == 1 ) {
    prior.probs <- rep(prior.probs, ncol(x));
  }

  n <- length(result$probne0);
  npred <- ncol(x);
  ndiff <- npred - n;
  bestindices <- (1:n)[result$which[1,]];
  pp100ind <- (1:n)[result$probne0 == 100];
  pp0ind <- (1:n)[result$probne0 == 0];
  if ( ndiff > 0 ) {
    pp0ind <- c(pp0ind, (n+1):npred);
  }
  minprob <- 0;
  if ( any(result$probne0 > 0) ) {
    minprob <- min(result$probne0[result$probne0 > 0])/100;
  }
  
  diffres <- BMA_Diff_BIC(y, as.matrix(x), prior.probs,
                          bestindices-1, diff100, diff0,
                          pp100ind-1, pp0ind-1, minprob, 1e-12)

  if ( ndiff > 0 ) {
    nmodels <- result$n.models;
  
    result$namesx <- colnames(x);
    result$which <- cbind(result$which, matrix(nrow=nmodels,
                                               ncol=ndiff, data=0));
    result$postmean <- c(result$postmean, rep(0, ndiff));
    result$postsd <- c(result$postsd, rep(0, ndiff));
    result$condpostmean <- c(result$condpostmean, rep(0, ndiff));
    result$condpostsd <- c(result$condpostsd, rep(0, ndiff));
    result$ols <- cbind(result$ols, matrix(nrow=nmodels,
                                           ncol=ndiff, data=0));
    result$mle <- cbind(result$mle, matrix(nrow=nmodels,
                                           ncol=ndiff, data=0));
    result$se <- cbind(result$se, matrix(nrow=nmodels,
                                         ncol=ndiff, data=0));
  }
  probne0 <- rep(0, npred);
  probne0[1:n] <- result$probne0;
  ak <- rep(0, npred);
  if ( diff100 ) {
    if (is.null(diffres$adjProb100)) stop("NULL100")
    diffres$adjProb100[is.na(diffres$adjProb100)] <- 100;
    ak[probne0==100] <- diffres$Ak100;
    probne0[probne0 == 100] <- diffres$adjProb100 * 100;
  }
  if ( diff0 ) {
    if (is.null(diffres$adjProb0)) stop("NULL0")
    diffres$adjProb0[is.na(diffres$adjProb0)] <- 0;
    ak[probne0==0] <- diffres$Ak0;
    probne0[probne0 == 0] <- diffres$adjProb0 * 100;
  }
  result$probne0 <- probne0;
  names(result$probne0) <- result$namesx;
  result$ak <- ak;
  names(result$ak) <- result$namesx;

  result;
}
