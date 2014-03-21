
ScanBMA <- function(x,
                    y,
                    prior.prob = NULL,
                    control = ScanBMAcontrol(),
                    verbose = FALSE) {

  ## For verbose mode, so don't have to check every time
  vprint <- function(...){;}
  if ( verbose ) {
    vprint <- function(...){print(...);}
  }
  
  if ( is.null(prior.prob) ) {
    prior.prob <- rep(.5, ncol(x));
  }
  if ( length(prior.prob) == 1 ) {
    prior.prob <- rep(prior.prob, ncol(x));
  }
  eps <- 1e-8;
  prior.prob = pmin( pmax( prior.prob, eps ), 1 - eps );
  if ( control$useg ) {
    vprint( "Running ScanBMA..." );
    result <- ScanBMA.g( x, y, prior.prob, control$OR, control$gCtrl, verbose );
    vprint( "Done" );
    class(result) <- c("ScanBMA","bicreg");
    result;
  }
  else {
    vprint( "Running ScanBMA..." );
    result <- ScanBMA.BIC( x, y, prior.prob, control$OR, verbose );
    vprint( "Done" );
    class(result) <- c("ScanBMA","bicreg");
    result
  }
}


