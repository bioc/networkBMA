ScanBMA_g <- function(y, x, priorProbs_, oddsRatio, g) {
    .Call('networkBMA_ScanBMA_g', PACKAGE = 'networkBMA', y, x, priorProbs_, oddsRatio, g)
}

ScanBMA_BIC <- function(y, x, priorProbs_, oddsRatio) {
    .Call('networkBMA_ScanBMA_BIC', PACKAGE = 'networkBMA', y, x, priorProbs_, oddsRatio)
}

BMA_Diff_g <- function(y, x, priorProbs_, g, bestModel, diff100, diff0, pred100, pred0, minprob, epsilon) {
  .Call('networkBMA_BMA_Diff_g', PACKAGE = 'networkBMA', y, x, priorProbs_, g, bestModel, diff100, diff0, pred100, pred0, minprob, epsilon)
}

BMA_Diff_BIC <- function(y, x, priorProbs_, bestModel, diff100, diff0, pred100, pred0, minprob, epsilon) {
  .Call('networkBMA_BMA_Diff_BIC', PACKAGE = 'networkBMA', y, x, priorProbs_, bestModel, diff100, diff0, pred100, pred0, minprob, epsilon)
}