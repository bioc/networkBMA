bicreg.pr <-
function(x, y, prior.prob = NULL, wt = rep(1, length(y)), strict = FALSE, 
         OR = 20, maxCol = 31, nbest = 10)
{

# leaps and Occam's window
# leaps does NOT take prior prob into consideration

    x <- as.matrix(x)
    xnames <- colnames(x)
    nvar <- ncol(x)

    maxCol <- min(nvar+1,maxCol)

    if (nvar > 2) {
        # no prior here
        a <- leaps(x, y, wt = wt, method = "r2", names = colnames(x), 
                   strictly.compatible = FALSE, nbest=nbest)
        a$r2 <- pmin(pmax(0, a$r2), 0.999)
        x.lm <- cbind.data.frame(y = y, as.data.frame(x[, a$which[2, 
            , drop = FALSE]]), w = wt)
        lm.fix <- lm(y ~ . - w, weights = wt, data = x.lm)
        r2.fix <- summary(lm.fix)$r.sq
        N <- ncol(x)
        magic <- N * log(1 - a$r2[2]) - N * log(1 - r2.fix)
        a$r2 <- 1 - (1 - a$r2) * exp(-magic/N)
        r2 <- round(c(0, a$r2) * 100, 3)
        size <- c(1, a$size)
        which <- rbind(rep(FALSE, ncol(x)), a$which)
    } else {
        # when there are only 2 variables
        r2 <- bic <- NULL
        nmod <- switch(ncol(x), 2, 4)
        bic <- rep(0, nmod)
        model.fits <- as.list(rep(0, nmod))
        which <- matrix(c(FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, TRUE),
                        nmod, nmod/2)
        size <- c(1, 2, 2, 3)[1:nmod]
        for (k in 1:nmod) {
            if (k == 1) {
                lm1 <- lm(y ~ 1, weights = wt)
            } else {
                x.lm <- cbind.data.frame(y = y, x = x[, which[k, 
                  , drop = FALSE]], wt = wt)
                lm1 <- lm(y ~ . - wt, data = x.lm, weights = wt)
            }
            r2[k] <- summary(lm1)$r.sq * 100
        }
    }
    # want to apply Occam's window in the very end (not 1st iteration)
    # for numerical reasons, remove models that are numerically impossible

    OR <- min(OR, .Machine$double.xmax)
    n <- length(y)
    bic <- n * log(1 - r2/100) + (size - 1) * log(n)
    postprob.crude.unrm <- -bic/2
    OR.fix <- 1

    if (!is.null(prior.prob)) {# Raftery 1995 paper
      # efficient version for the multiplication term over the regulators
      # in each model M_k, compute prior Pr(Mk)
        prior.mprob <- colSums( apply(which, 1, ifelse, 
           yes=log(prior.prob+.Machine$double.neg.eps), 
           no=log((1-prior.prob)+.Machine$double.neg.eps) ))
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

    # strict / symmetric Occam's Window
    # strict = FALSE by default
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
        }
    }

    nmod <- length(bic)
    postprob <- rep(0, nmod)
    model.fits <- matrix(0, nmod, maxCol)
    for (k in 1:nmod) {
        if (sum(which[k, ]) != 0) {
            res.lm <- lm(y~x[,which[k,],drop=F])
            # relatively accurate approx to postprob, not with BIC
            # logLik = maximized LOG likelihood
            model.fits[k, c(T,which[k,])] <- coef(res.lm)
        } else {
            res.lm <- lm(y~1)
            model.fits[k, 1] <- coef(res.lm)
        }
        postprob[k] <- logLik(res.lm) - (size[k]-1)*log(n)
    }
    
    # compute the model's posterior probability
    # up to here, postprob does NOT incorporate prior prob
    score <- BIC <- -2*postprob
    if (!is.null(prior.prob)) {
        # compute Pr(Mk|D) proportional to log likelihood * Pr(Mk)
        # working in log space, so + instead of *
        postprob <- postprob + prior.mprob
        score <- BIC - 2*prior.mprob
    } 

# Adrian
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
    # undo the LOG for likelihood
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
# postprob.crude.unrm: not normalized version
# needed at the end when we normalize everything
    result <- list(bic = BIC, 
                   postprob = postprob, priorprob = priorprob, 
                   namesx = xnames, label = label, r2 = r2, size = size, 
                   which = which, probne0 = probne0 * 100, 
                   postmean = Ebi, condpostmean = CEbi,
                   ols = model.fits, mle = model.fits,
                   n.models = nmod, n.vars = nvar,
                   postprob.crude.unrm = postprob.crude.unrm, 
                   postprob.crude = postprob.crude)
    # new class, distinct from bicreg
#   class(result) <- "bicreg.pr"
    result
}
