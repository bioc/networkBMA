ScanBMA.g <- function(x,
                      y,
                      prior.probs = rep(.5, ncol(x)),
                      OR = 100,
                      gCtrl = gControl(),
                      verbose = FALSE ) {

  ## For verbose mode, so don't have to check every time
  vprint <- function(...){;}
  if ( verbose ) {
    vprint <- function(...){print(...);}
  }
  
  delta <- 10;
  counter <- 0;
  
  n <- length(y);
  gcur <- sqrt(n);
  if ( !is.null(gCtrl$g0) ) {
    gcur <- gCtrl$g0;
  }
  res <- list(edgelist=matrix(NA, nrow=1, ncol=3),
              modelprobs=NULL, sizes=NULL, r2s=NULL);
  bmaresult <- NULL;
  if ( gCtrl$optimize ) {
    while ( delta > gCtrl$epsilon &
           counter < gCtrl$iterlim ) {
      bmaresult <- FastScanBMA.g( x=x, y=y, prior.probs=prior.probs,
                                 OR=OR, g=gcur );
      res$modelprobs <- bmaresult$postprob;
      res$sizes <- bmaresult$size;
      res$r2s <- bmaresult$r2/100;
      gnew <- Optimizeg( res$modelprobs, res$sizes, res$r2s, n, gcur )$par;
      delta <- abs(gcur - gnew);
      gcur <- gnew;
      counter <- counter + 1;
      vprint(paste("Optimizing g: iteration ", counter, ", g = ", gcur, sep=""));
    }
    if ( counter == gCtrl$iterlim ) {
      print("too many iterations for g optimization");
    }
  }
  else {
    bmaresult <- FastScanBMA.g( x=x, y=y, prior.probs=prior.probs,
                               OR=OR, g=gcur );
  }
  bmaresult$g <- gcur;
  bmaresult;
}

 # function for finding the optimum value of g
 # for the current set of model probabilities, sizes
 # and R^2s
 Optimizeg <- function( modelprobs, modelsizes, r2s, n, g0 ) {
   optimfun <- function(g) {
     if ( g < 1 | g > n ) {
       10^10;
     }
     else {
       - sum( modelprobs * ( (n - modelsizes - 1)*log(1+g) - (n-1)*log(1+g*(1-r2s)) ) );
     }
   }
   optim( c(g0), optimfun, method="Brent", lower=1, upper=n );
 }

 BMA.Diff.g <- function( x, y, prior.probs, result, diff100, diff0 ) {
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
   if ( length(result$probne0) > 0 && any(result$probne0 > 0) ) {
     minprob <- min(result$probne0[result$probne0 > 0])/100;
   }

   diffres <- BMA_Diff_g(y, as.matrix(x), prior.probs, result$g,
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
     diffres$adjProb100[is.na(diffres$adjProb100)] <- 100;
     ak[probne0==100] <- diffres$Ak100;
     probne0[probne0 == 100] <- diffres$adjProb100 * 100;
   }
   if ( diff0 ) {
     diffres$adjProb0[is.na(diffres$adjProb0)] <- 0;
     ak[probne0==0] <- diffres$Ak0;
     probne0[probne0 == 0] <- diffres$adjProb0 * 100;
   }
   result$probne0 <- probne0;
   names(result$probne0) <- result$namesx;
   result$ak <- ak;
   names(result$ak) <- result$namesx;
   result$ak[result$ak == Inf] <- 1e5;

   result;
 }
