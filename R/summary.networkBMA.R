"summary.networkBMA" <- function(object, thresholds = c(0,.5,.75,.9,.95,1),
                                                                      ...) {
  s <- sapply( thresholds, function(p) sum(object[,3] >= p))
  names(s) <- paste(thresholds * 100, "%", sep = "")
  s  
}
