bicreg.pr.resid <-
function(x, y, known, prior.prob = NULL, wt = rep(1, length(y)), 
         strict = FALSE, OR = 20, maxCol = 31, nbest = 10)
{

    nKnown <- ncol(known$x)

    # leaps and Occam's window
    xnames <- colnames(known$residuals$x)
    nvar <- ncol(known$residuals$x)

    maxCol <- min(nvar+1,maxCol)

    if (nvar > 2) {
        a <- leaps(known$residuals$x, known$residuals$y, wt = wt, method = "r2", 
   names = colnames(known$residuals$x), strictly.compatible = FALSE, nbest=nbest)
        a$r2 <- pmin(pmax(0, a$r2), 0.999)
        x.lm <- cbind.data.frame(y = known$residuals$y, 
        as.data.frame(known$residuals$x[, a$which[2, , drop = FALSE]]), w = wt)
        lm.fix <- lm(y ~ . - w, weights = wt, data = x.lm)
        r2.fix <- summary(lm.fix)$r.sq
        N <- ncol(known$residuals$x)
        magic <- N * log(1 - a$r2[2]) - N * log(1 - r2.fix)
        a$r2 <- 1 - (1 - a$r2) * exp(-magic/N)
        r2 <- round(c(0, a$r2) * 100, 3)
        size <- c(1, a$size)
        which <- rbind(rep(FALSE, ncol(known$residuals$x)), a$which)
    } else {
        r2 <- bic <- NULL
        nmod <- switch(ncol(known$residuals$x), 2, 4)
        bic <- rep(0, nmod)
        model.fits <- as.list(rep(0, nmod))
        which <- matrix(c(FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, TRUE),
                          nmod, nmod/2)
        size <- c(1, 2, 2, 3)[1:nmod]
        for (k in 1:nmod) {
            if (k == 1) {
                lm1 <- lm(known$residuals$y ~ 1, weights = wt)
            } else {
                x.lm <- cbind.data.frame(y = known$residuals$y, 
                x = known$residuals$x[, which[k, ,drop = FALSE]], wt = wt)
                lm1 <- lm(y ~ . - wt, data = x.lm, weights = wt)
            }
            r2[k] <- summary(lm1)$r.sq * 100
        }
    }

    OR <- min(OR, .Machine$double.xmax)
    n <- length(known$residuals$y)
    bic <- n * log(1 - r2/100) + (size - 1) * log(n)
    if (any(is.na(bic))) stop("BIC")
    postprob.crude.unrm <- -bic/2
    OR.fix <- 1 
    if (!is.null(prior.prob)) {
        prior.mprob <- colSums( apply(which, 1, ifelse, 
            yes=log(prior.prob+.Machine$double.neg.eps), 
                 no=log((1-prior.prob)+.Machine$double.neg.eps) ))
        if (any(is.na(prior.mprob))) {
          print(range(as.numeric(which)))
          print(range(prior.prob))
          stop("prior")
        }
        postprob.crude <- -bic/2 + prior.mprob
        occam <- postprob.crude - max(postprob.crude) > -OR.fix * log(OR)
        prior.mprob <- prior.mprob[occam]
    } else {
        postprob.crude <- -bic/2
        occam <- bic - min(bic) < 2 * OR.fix * log(OR)
    }
    r2 <- r2[occam]
    size <- size[occam]
    which <- which[occam, , drop = FALSE]
    bic <- bic[occam]
    postprob.crude.unrm <- postprob.crude.unrm[occam]
    postprob.crude <- postprob.crude[occam]

     if (any(is.na(which))) {
          print(dim(which))
          print(length(occam))
          print(sum(is.na(occam)))
           stop("which")
      }

    # strict / symmetric Occam's Window
    order.bic <- order(-postprob.crude, size)
    if (!is.null(prior.prob))
        prior.mprob <- prior.mprob[order.bic]
    r2 <- r2[order.bic]
    size <- size[order.bic]
    which <- which[order.bic, , drop = FALSE]
    bic <- bic[order.bic]
    postprob.crude.unrm <- postprob.crude.unrm[order.bic]
    postprob.crude <- postprob.crude[order.bic]

    if (strict) {
        nmod <- length(bic)
        if (nmod > 1) {
            occam <- rep(TRUE, nmod)
            for (k in (2:nmod)) {
                for (j in (1:(k - 1))) {
                    which.diff <- which[k, ] - which[j, ]
                    if (all(which.diff >= 0)) 
                        occam[k] <- FALSE
                }
            }
            if (!is.null(prior.prob))
                prior.mprob <- prior.mprob[occam]
            r2 <- r2[occam]
            size <- size[occam]
            nmod <- sum(occam)
            which <- which[occam, , drop = FALSE]
            bic <- bic[occam]
            postprob.crude.unrm <- postprob.crude.unrm[occam]
            postprob.crude <- postprob.crude[occam]
            #postprob <- postprob/sum(postprob)
        }
    }

    xnames <- c(colnames(known$x), colnames(known$residuals$x))
    nvar <- length(xnames)
#  moved
#   which <- cbind(matrix(T, length(bic), nKnown), which)
    size <- nKnown + size
    nmod <- length(bic)
    postprob <- rep(0, nmod)
    model.fits <- matrix(0, nmod, nKnown+maxCol)
    for (k in 1:nmod) {
        if (sum(which[k, ]) != 0) {
            res.lm <- lm(y~cbind(known$x, x[ , which[k,], drop=FALSE]))
            model.fits[k, c(rep(T,nKnown+1),which[k,])] <- coef(res.lm)
        } else {
            res.lm <- lm(y~known$x)
            model.fits[k, 1:(nKnown+1)] <- coef(res.lm)
        }
        postprob[k] <- logLik(res.lm) - (size[k]-1)*log(n)/2
    }

    which <- cbind(matrix(T, length(bic), nKnown), which)
    
    # compute the model's posterior probability
    score <- BIC <- -2*postprob
    if (!is.null(prior.prob)) {
        postprob <- postprob + prior.mprob
        score <- BIC - 2*prior.mprob
    } 

#   occam <- postprob - max(postprob) > -log(OR)
    occam <- score - min(score) < 2*log(OR)
    if (!is.null(prior.prob)) {
        prior.mprob <- prior.mprob[occam]
        priorprob <- exp(prior.mprob-min(prior.mprob)) / 
                          sum(exp(prior.mprob-min(prior.mprob)))
    } else
        priorprob <- NULL
    r2 <- r2[occam]
    size <- size[occam]
    which <- which[occam, , drop = FALSE]
    bic <- bic[occam]
    postprob.crude.unrm <- postprob.crude.unrm[occam]
    postprob.crude <- postprob.crude[occam]
    postprob.crude <- postprob.crude - min(postprob.crude)
    postprob.crude <- exp(postprob.crude) / sum(exp(postprob.crude))
    BIC <- BIC[occam]
    postprob <- postprob[occam]
    postprob <- postprob - min(postprob)
    postprob <- exp(postprob) / sum(exp(postprob))
    model.fits <- model.fits[occam, , drop=F]
    nmod <- length(bic)

    probne0 <- c(t(which) %*% postprob)
    Ebi <- c(t(model.fits) %*% postprob)
    CEbi <- Ebi/c(1,probne0)
    colnames(which) <- names(probne0) <- xnames
    names(Ebi) <- names(CEbi) <- colnames(model.fits) <- c("Int", xnames)
    templabs <- t(matrix(rep(colnames(which), times = nrow(which)), 
        ncol = nrow(which)))
    templabs[!which] <- ""
    label <- apply(templabs, 1, paste, collapse = "")
    label[label==""] <- "NULL"
    result <- list(bic = BIC, 
                   postprob = postprob, priorprob = priorprob, 
                   namesx = xnames, label = label, r2 = r2, size = size, 
                   which = which, probne0 = probne0 * 100, 
                   postmean = Ebi, condpostmean = CEbi,
                   ols = model.fits, mle = model.fits,
                   n.models = nmod, n.vars = nvar,
                   postprob.crude.unrm = postprob.crude.unrm, 
                   postprob.crude = postprob.crude)
#   class(result) <- "bicreg.pr"
    result
}

