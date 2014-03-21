## Parameters:
# y - the target data (dependent variable) as a column vector
# x - the H predictors (independent variables) as an n x H matrix
# prior.probs - the prior probabilities of the predictors
# OR - the odds ratio to use to define our window of acceptable models
## Returns:
# an object of class bicreg
FastScanBMA.BIC <- function( x, y, prior.probs, OR=100 ) {
  cl <- match.call();
  x <- data.frame(x);
  if (is.null(dimnames(x))) {
    dimnames(x) <- list(NULL, paste("X", 1:ncol(x), sep = ""));
  }
  y <- as.numeric(y);
  xnames <- dimnames(x)[[2]];
  nvar <- length(x[1, ]);

  # Call the fast scanBMA rcpp function
  rcpp.results <- ScanBMA_BIC( y, as.matrix(x), prior.probs, OR );

  modelR2s <- round(rcpp.results$r2, 3);
  modelBICs <- rcpp.results$bic;

  which <- as.matrix(rcpp.results$which == 1);
  dimnames(which) <- list(NULL, xnames);
  size <- apply(which, 1, sum);
  
  n.models <- length(modelBICs);
  if ( n.models == 0 ) {
    result <- list(postprob = NULL, 
                   namesx = NULL,
                   label = NULL, 
                   r2 = NULL, 
                   bic = NULL, 
                   size = NULL, 
                   which = NULL, 
                   probne0 = NULL, 
                   postmean = NULL, 
                   postsd = NULL, 
                   condpostmean = NULL, 
                   condpostsd = NULL, 
                   ols = NULL, 
                   mle = NULL, 
                   se = NULL, 
                   reduced = FALSE, 
                   dropped = NULL, 
                   call = NULL, 
                   n.models = n.models,
                   n.vars = 0,
                   nmodelschecked = rcpp.results$nmodelschecked
                   );
    class(result) <- "bicreg";
    return(result);
  }
  labels <- apply(which, 1, function(x){paste(xnames[x], collapse="");});
  
  # Calculate Posterior Probabilities
  postprob <- rcpp.results$postprob;
  
  # Probability not equal to zero
  probne0 <- round(rcpp.results$probne0, 5);
  names(probne0) <- xnames;

  ##############
  nmod <- n.models;
  label <- labels;
  bic <- modelBICs;
  # Copied straight from bicreg
  model.fits <- as.list(rep(0, nmod))
  for (k in (1:nmod)) {
      if (sum(which[k, ]) != 0) {
          model.fits[[k]] <- ls.print(lsfit(x[, which[k, ], 
              drop = FALSE], y), print.it = FALSE)$coef.table[[1]]
      }
      else model.fits[[k]] <- ls.print(lsfit(rep(1, length(y)), 
          y, intercept = FALSE), print.it = FALSE)$coef.table[[1]]
  }
  Ebi <- rep(0, (nvar + 1))
  SDbi <- rep(0, (nvar + 1))
  CEbi <- Ebi
  CSDbi <- SDbi
  EbiMk <- matrix(rep(0, nmod * (nvar + 1)), nrow = nmod)
  sebiMk <- matrix(rep(0, nmod * (nvar + 1)), nrow = nmod)
  for (i in 1:(nvar + 1)) {
      if ((i == 1) || (sum(which[, (i - 1)] != 0))) {
          for (k in (1:nmod)) {
              if ((i == 1) || (which[k, (i - 1)] == TRUE)) {
                if (i == 1) 
                  pos <- 1
                else pos <- 1 + sum(which[k, (1:(i - 1))])
                EbiMk[k, i] <- model.fits[[k]][pos, 1]
                sebiMk[k, i] <- model.fits[[k]][pos, 2]
              }
          }
          Ebi[i] <- as.numeric(sum(postprob * EbiMk[, i]))
          SDbi[i] <- sqrt(postprob %*% (sebiMk[, i]^2) + postprob %*% 
              ((EbiMk[, i] - Ebi[i])^2))
          if (i == 1) {
              CEbi[i] <- Ebi[i]
              CSDbi[i] <- SDbi[i]
          }
          else {
              sel <- which[, i - 1]
              cpp <- postprob[sel]/sum(postprob[sel])
              CEbi[i] <- as.numeric(sum(cpp * EbiMk[sel, i]))
              CSDbi[i] <- sqrt(cpp %*% (sebiMk[sel, i]^2) + 
                cpp %*% ((EbiMk[sel, i] - CEbi[i])^2))
          }
      }
  }
  dimnames(EbiMk) <- dimnames(sebiMk) <- list(NULL, c("Int", 
      dimnames(x)[[2]]))
  ##############
  
  result <- list(postprob = postprob, 
                 namesx = xnames,
                 label = labels, 
                 r2 = modelR2s, 
                 bic = modelBICs, 
                 size = size, 
                 which = which, 
                 probne0 = c(probne0), 
                 postmean = Ebi, 
                 postsd = SDbi, 
                 condpostmean = CEbi, 
                 condpostsd = CSDbi, 
                 ols = EbiMk, 
                 mle = EbiMk, 
                 se = sebiMk, 
                 reduced = FALSE, 
                 dropped = NULL, 
                 call = cl, 
                 n.models = n.models,
                 n.vars = length(probne0),
                 nmodelschecked = rcpp.results$nmodelschecked
                 );
  class(result) <- "bicreg";
  result;
}
