"summary.networkBMA" <- function(object, threshholds = c(0,.5,.75,.9,.95,1),
                                                                      ...) {
  s <- sapply( threshholds, function(p) sum(object[,3] >= p))
  names(s) <- paste(threshholds * 100, "%", sep = "")
  s  
}
