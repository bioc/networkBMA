iterateBMAlm <-
function(x, y, prior.prob = NULL, control = iBMAcontrolLM(), verbose = FALSE) {

    # Get the top "maxNvar" variables (replaces call to iterateBMAinit)

    ## if (!exists("leaps")) library(BMA)

    maxNvar <- min(control$maxNvar, ncol(x))
 
    IN <- 1:maxNvar
    nextVar <- maxNvar + 1

    # Iterative bicreg
    currIter <- 0
    ret.bic.reg <- vector("list", ncol(x)-maxNvar+1)
    while (currIter < control$maxIter) {

        currIter <- currIter + 1

        # Run bicreg
        if (verbose) {
          cat(paste("Current iteration is ", currIter, "\n", sep = ""))
          cat("Apply bicreg now\n")
        }

        ret.bicreg <- bicreg.pr(x[,IN,drop=FALSE], y, 
                 prior.prob=if(!is.null(prior.prob)) prior.prob[IN] else NULL, 
                                OR=if(control$keepModels) Inf else control$OR, 
                                nbest=control$nbest, maxCol=control$maxNvar+1)

        probne0 <- ret.bicreg$probne0

        if (control$keepModels) ret.bic.reg[[currIter]] <- ret.bicreg

        if (verbose) {
            cat("After bicreg\n")
        }
        # Get a logical vector for which probne0 < thresProbne0, i.e., get a vector of variables whose probne0's are too low
        rmVector <- which(probne0 < control$thresProbne0)
        if (verbose) {
            cat("Posterior Probabilities of selected genes:\n")
            print(probne0)
            cat(paste("Length of rmVector is ", length(rmVector), "\n", sep = ""))
        }

        if (verbose) cat(currIter, " ")

        if (length(rmVector) == 0) {
            # No gene to swap in!! Increase threshold
            # also make sure that not all probne0 = 100%
            currMin <- min(probne0)
            if (verbose ==TRUE) {
                cat (paste("No gene to swap! Min probne0 = ", currMin, "\n", sep=""))
            }

            # adaptive threshold: Assign new threshold
            if (currMin < 99) {
                newThresProbne0 <- currMin + 1
                rmVector <- which(probne0 < newThresProbne0)
                if (verbose == TRUE) {
                    cat(paste("New probne0 threshold = ", newThresProbne0, "\n", sep=""))
                    cat("New rmVector after applying increased threshold:\n")
                    print(rmVector)
                }
            } else {
                break
            }
        }

        # Now, there is at least 1 gene to swap, guaranteed
        if (nextVar <= ncol(x)) {
            # Set up new X
            if (verbose == TRUE) {
                cat("Set up new X\n")
                cat(paste("nextVar is ", nextVar, "\n", sep=""))
            }
            lastVar <- length(rmVector) + nextVar - 1  
            # Make sure lastVar <= ncol(x)
            if (lastVar > ncol(x)) {
                rmVector <- rmVector[1:(ncol(x) -nextVar + 1)]
                lastVar <- ncol(x)
            }
            # Update curr.mat and curr.prior.prob
            IN <- c(IN[-rmVector],nextVar:lastVar)
            nextVar <- lastVar + 1
        } else {
            # There is no variable to be removed OR exhausted all data
            break
        }
    }

    if (verbose) cat(paste(currIter, ": Explored up to variable # ", nextVar-1, "\n", sep=""))
    # Print out selected genes if iterateBMAlm has completed
    if (verbose && currIter < control$maxIter) {
        cat ("Iterate bicreg is done OR all probne0 are 100%!\n")
    }

    ##############################################
    # Occam's window: to account for all models selected in ALL iterations
    # use a list object to store results from each iterations
    # these list objects are memory inefficient, may want to consider loops

if (control$keepModels) {
    length(ret.bic.reg) <- currIter
    BIC <- unlist(sapply(ret.bic.reg, "[[", i="bic"))
    postprob.crude.unrm <- unlist(sapply(ret.bic.reg, "[[", i="postprob.crude.unrm"))
    xnames <- colnames(x)
    label <- unlist(sapply(ret.bic.reg, "[[", i="label"))
    r2 <- unlist(sapply(ret.bic.reg, "[[", i="r2"))
    size <- unlist(sapply(ret.bic.reg, "[[", i="size"))
    nvar <- ncol(x)

    which <- matrix(FALSE, length(BIC), ncol(x))
    colnames(which) <- colnames(x)
    model.fits <- matrix(0, length(BIC), ncol(x)+1)
    colnames(model.fits) <- c("Int", colnames(x))
    stop.ind <- 0
    for (i in 1:length(ret.bic.reg)) {
        start.ind <- stop.ind + 1
        stop.ind <- stop.ind + nrow(ret.bic.reg[[i]]$which)
        which[start.ind:stop.ind, ret.bic.reg[[i]]$namesx] <- ret.bic.reg[[i]]$which
        model.fits[start.ind:stop.ind, c("Int", ret.bic.reg[[i]]$namesx)] <- ret.bic.reg[[i]]$mle
    }
    # augment window to contain all candidate models considered
    if (!is.null(prior.prob)) {
        # compute prior Pr(Mk)
        prior.mprob <- colSums( apply(which, 1, ifelse, 
         yes=log(prior.prob+.Machine$double.neg.eps), 
         no=log((1-prior.prob)+.Machine$double.neg.eps) ))
        postprob.crude <- postprob.crude.unrm + prior.mprob
        postprob <- -BIC/2 + prior.mprob
    } else {
        postprob.crude <- postprob.crude.unrm
        postprob <- -BIC/2
    }

    # apply Occam's window, OR=20 after all iterations of iBMA
    # in the iterative step, we set OR=Inf to avoid eliminating any models until now
    occam <- postprob - max(postprob) > -log(control$OR)
    if (!is.null(prior.prob)) prior.mprob <- prior.mprob[occam]
    r2 <- r2[occam]
    size <- size[occam]
    which <- which[occam, , drop = FALSE]
    postprob.crude.unrm <- postprob.crude.unrm[occam]
    postprob.crude <- postprob.crude[occam]
    BIC <- BIC[occam]
    postprob <- postprob[occam]
    model.fits <- model.fits[occam, , drop=F]
    label <- label[occam]

    # sort models in desc order of postprob
    order.bic <- order(-postprob, size, label)
    if (!is.null(prior.prob)) prior.mprob <- prior.mprob[order.bic]
    r2 <- r2[order.bic]
    size <- size[order.bic]
    which <- which[order.bic, , drop = FALSE]
    postprob.crude.unrm <- postprob.crude.unrm[order.bic]
    postprob.crude <- postprob.crude[order.bic]
    BIC <- BIC[order.bic]
    postprob <- postprob[order.bic]
    model.fits <- model.fits[order.bic, , drop=F]
    label <- label[order.bic]

    # to remove duplicated models returned from different iterations
    # by checking the labels of models
    inc <- c(T, label[-1]!=label[-length(label)])
    if (!is.null(prior.prob)) {
        prior.mprob <- prior.mprob[inc]
        priorprob <- exp(prior.mprob-min(prior.mprob)) / sum(exp(prior.mprob-min(prior.mprob)))
    } else
        priorprob <- NULL
    r2 <- r2[inc]
    size <- size[inc]
    which <- which[inc, , drop = FALSE]
    postprob.crude.unrm <- postprob.crude.unrm[inc]
    postprob.crude <- postprob.crude[inc]
    postprob.crude <- postprob.crude - min(postprob.crude)
    postprob.crude <- exp(postprob.crude) / sum(exp(postprob.crude))
    BIC <- BIC[inc]
    postprob <- postprob[inc]
    postprob <- postprob - min(postprob)
    postprob <- exp(postprob) / sum(exp(postprob))
    model.fits <- model.fits[inc, , drop=F]
    label <- label[inc]
    nmod <- length(BIC)

    probne0 <- c(t(which) %*% postprob)
    Ebi <- c(t(model.fits) %*% postprob)
    CEbi <- Ebi/c(1,probne0)
    names(probne0) <- xnames
    names(Ebi) <- names(CEbi) <- colnames(model.fits) <- c("Int", xnames)

    obj <- list(bic = BIC, postprob = postprob, 
                priorprob = priorprob, namesx = xnames, label = label, r2 = r2, 
                size = size, which = which, probne0 = probne0 * 100, 
                postmean = Ebi, condpostmean = CEbi, ols = model.fits, 
                mle = model.fits, n.models = nmod, n.vars = nvar)
   }
   else obj <- ret.bicreg

#   ret.val <- c(obj, list(curr.names = colnames(x)))
#   class(ret.val) <- "bicreg.pr"
#   ret.val <- list(obj=obj, curr.names=colnames(x))

    if (verbose) cat("Selected genes:\n")
    curr.ind <- which(obj$probne0 > 0)
    if (verbose) print(obj$namesx[curr.ind])
    if (verbose) cat("Posterior probabilities of selected genes:\n")
    if (verbose) print(obj$probne0[curr.ind])
#   ret.val
    obj
}

