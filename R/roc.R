roc <-
function (contabs, plotit = TRUE) 
{
    legend <- FALSE
    auc <- function(x, y) {
        nBlocks <- length(y) - 1
        rec <- tri <- rep(0, nBlocks)
        i <- 1
        for (j in 2:length(y)) {
            xdiff <- (x[j] - x[i])
            rec[i] <- xdiff * min(y[c(i, j)])
            tri[i] <- (xdiff * abs(y[j] - y[i]))
            i <- j
        }
        list(block = sum(rec), trapezoid = sum(rec) + sum(tri)/2)
    }
    scr <- scores(contabs, what = c("FPR", "TPR"))
    ord <- order(scr$FPR, scr$TPR)
    xx <- x <- scr$FPR[ord]
    l <- length(xx)
    width <- diff(range(x))
    yy <- y <- scr$TPR[ord]
#   area <- sectorArea <- auc(x, y)$trap
    area <- sectorArea <- auc(x, y)$block
    if (min(xx) > 0) {
      x <- c(0,x)
      y <- c(0,y)
      triangle <- (xx[1]*yy[1])/2
      area <- area + triangle
    }
    if (max(xx) < 1) {
      x <- c(x,1)
      y <- c(y,1)
      trapezoid <- ((1-xx[l])*yy[l]) + ((1-xx[l])*(1-yy[l]))/2
      area <- area + trapezoid
    }
    if (plotit) {
        plot(xx, yy, type = "n", xlim = c(0,1), ylim = c(0,1), 
            xlab = "FPR (1 - specificity)", 
            ylab = "TPR (sensitivity)", lty = 1)
        abline( v = range(xx), col = "red", lty = 1)
        lines(xx,yy)
        if (min(xx) > 0) segments(0, 0, xx[1], yy[1], lty = 2)
        if (max(xx) < 1) segments(xx[l], yy[l], 1, 1, lty = 2)
        lines( c(0,1), c(0,1), lty = 1, col = "blue")
        if (legend) {
            legend("bottomright", legend = c(paste("area", round(area, 
                3), sep = " = ")))
        }
    }
   c( area = area, sector = sectorArea, width = width)
}

