prc <-
function (contabs, plotit = TRUE, ymax = max(y)) 
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
    
    P <- unique( contabs$TP + contabs$FN)
    N <- unique( contabs$FP + contabs$TN)
    if (length(P) > 1 || length(N) > 1) stop("problem with contingency tables")

    pr <- scores(contabs, what = c("TP", "FP", "precision", "recall"))
    pr$precision[pr$TP == 0 & pr$FP == 0] <- 1
    pr <- pr[order(pr$recall, pr$precision),]
    splitPR <- split( pr, pr$recall)
    pr <- pr[!duplicated(pr$recall),]

    width <- diff(range(pr$recall))
    xx <- pr$recall
    yy <- pr$precision
#   sectorArea <- auc(xx, yy)$trap
    sectorArea <- auc(xx, yy)$block

    if (!any(pr$TP == 0)) {
      pr <- rbind(c(TP = 0, FP = P, precision = 0, recall = 0), pr)
    }
    if (!any(pr$TP == P)) {
      pr <- rbind(pr,c(TP = P, FP = N, precision = (P/(P+N)), recall = 1))
    }

    rownames(pr) <- 1:nrow(pr)

    TP <- seq( from = 0, to = P, by = 1)
    PR <- data.frame( precision = rep(-1,length(TP)), 
                      recall = rep(-1,length(TP)),
                      TP = TP)
    k <- 0
    i <- 1
    TPi <- pr[1,]$TP
    for (j in 2:nrow(pr)) {
         TPj <- pr[j,]$TP
         dTPij <- TPj - TPi
         skew <- (pr[j,]$FP - pr[i,]$FP)/dTPij
         FPi <- pr[i,]$FP
         for (l in (1:dTPij)-1) {
            tp <- TPi + l
            k <- k + 1
            PR[k, ]$recall <- tp/P
            fp <- FPi + skew*l
            if (tp == 0 && fp == 0) {
              PR[k, ]$precision <- 1
            }
            else {
              PR[k, ]$precision <- tp/(tp + fp)
            }  
         }
       i <- j
       TPi <- TPj
    }
    k <- k + 1
    PR[k, ]$recall <- pr[nrow(pr),]$recall
    PR[k, ]$precision <- pr[nrow(pr),]$precision

    rownames(PR) <- 1:nrow(PR)

    x <- PR$recall
    y <- PR$precision
#   area <- auc(x, y)$trap
    area <- auc(x, y)$block

    ymax <- min(ymax,1)
    ylim <- c(0, ymax)
    if (plotit) {
        plot(x, y, ylim = ylim, type = "n", xlab = "Recall", 
            ylab = "Precision")
        abline( v = range(xx), col = "red", lty = 1)
        y <- sapply(sort(unique(x)), function(z, x, y) max(y[x == 
            z]), x = x, y = y)
        x <- sort(unique(x))
        for (i in 2:length(x)) {
            IN <- c((i - 1), i)
#           ymin <- min(y[IN])
            sapply( splitPR, function(z) {
# spikes
                if (nrow(z) > 1) {
                  recall <- z[1,]$recall
                  segments( recall, min(z$precision), 
                            recall, max(z$precision), lty = 1)
                }
            })
            LTY <- 1
            if (x[i] <= min(xx)) LTY <- 2
            if (x[i-1] >= max(xx)) LTY <- 2
            segments(x[i - 1], y[i - 1], x[i], y[i], lty = LTY)
#           ymax <- max(y[IN])
#           if (ymax != ymin) {
#               xmax <- x[IN][y[IN] == ymax]
#               segments(xmax, ymin, xmax, ymax, lty = 1)
#            }
        }
        if (legend) {
            legend("topright", legend = c(paste("area", round(area, 
                3), sep = " = ")))
        }
    }
    c( area = area, sector = sectorArea, width = width)
}
