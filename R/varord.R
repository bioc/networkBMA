"varord" <- function(x, y, prior.prob = NULL, 
                     ordering = c("bic1","prior","bic1+prior")) {

# this assumes that the columns of x are the variables (genes)

if (is.null(prior.prob)) prior.prob <- rep(1,ncol(x))

switch( ordering[1],
   "prior" = {
            x <- t(x)
            p <- nrow(x)
            gene.rank <- rank(-prior.prob, ties.method="min")
            x <- x[gene.rank<=p,]
            prior.prob <- prior.prob[gene.rank<=p]
            gene.rank <- gene.rank[gene.rank<=p]
            ties.rank <- as.numeric(names(which(table(gene.rank)>1)))
            ties.rank.ind <- gene.rank %in% ties.rank
            abs.corr <- rep(NA, length(prior.prob))
            if (any(ties.rank.ind))  abs.corr[ties.rank.ind] <-
              abs(cor(t(x[ties.rank.ind,,drop=F]), y, "pairwise"))
            order(-gene.rank, abs.corr, decreasing=T)
           },
   "bic1" = {
            bic1fun <- function(x,y) -2*logLik(lm(y ~ x)) + 2*log(length(y))
            score <- apply( x, 2, bic1fun, y = y)
            order( score, decreasing = FALSE)
           },
   "bic1+prior" = {
           logodds <- function(prob, eps = .Machine$double.neg.eps) {
                               log(prob+eps) - log((1-prob)+eps)
                              }
            bic1fun <- function(x,y) -2*logLik(lm(y ~ x)) + 2*log(length(y))
            score <- apply( x, 2, bic1fun, y = y) - 2*logodds(prior.prob)
            order( score, decreasing = FALSE)
           },
    stop("unrecognized ordering")
   )
}
