\name{networkBMA}
\alias{networkBMA}
\title{
   Gene network inference from time series data via BMA.
}
\description{
  Estimates a probabilistic network of regulator-gene pairs   
  from time series data via iterative BMA.
}
\usage{
networkBMA(data, nTimePoints, prior.prob = NULL, known = NULL, 
           ordering = "bic1+prior", nvar = NULL, self = TRUE, 
           maxreg = NULL, control = iBMAcontrolLM(thresProbne0 = 5), 
           verbose = FALSE) 
}
\arguments{
  \item{data}{
    A matrix whose columns correspond to variables or genes and whose
    rows correspond to the observations at different time points.
    The row names should be the gene names, to used in conjunction with 
   \code{prob} and \code{known} as appropriate.
  }
  \item{nTimePoints}{
    The number of time points at which expression measurements are available.
    The number of columns in \code{data} should be a multiple of 
    \code{nTimePoints}, which could be greater than 1 if there are replicates.
   }
 \item{prior.prob}{
     If included as input, either a single positive fraction representing the 
     probability of an regulator-gene pair in the network, or else a 
     matrix in which the (i,j) entry is the estimated prior probability that 
     gene i regulates gene j. The default value is \code{NULL}, which implies
     that no prior information will be used in modeling the network.
 }
 \item{known}{
     An optional 2-column matrix of known (hard-coded) regulatory 
     relationships. The first column gives the name of the regulator, 
     and the second column gives the name of the target gene. 
     The gene names should be consistent with the data.
  }
 \item{ordering}{
     A character string indicating the ordering to be used for the 
     genes or variables, referring to the options for ordering in function 
     \code{varord}. The default is option \code{"bic1+prior"} for 
     \code{varord}. \cr
     If a prior is provided, genes with \code{prob=1} are always included
     first in modeling regardless of the specified ordering, while genes with
     \code{prior = 0} are excluded.
}
 \item{nvar}{
     The number of top-ranked (see \code{ordering}) genes to be considered in 
     the modeling. 
     The default is determined by the length of \code{ordering} if it
     is greater than 1, otherwise it is \code{min(nrow(data),ncol(data)-1)}.
     The maximum number of genes for which measurements are available is 
     \code{nrow(data)}.
 }
  \item{self}{
     A logical variable indicating whether or not to allow self edges in
     modeling. The default is to allow self edges.
  }
 \item{maxreg}{
     An optional estimate of the maximum number of regulators for any
     gene in the network. If provided, this is used to help reduce the 
     amount of memory used for the computations.
  }
  \item{control}{
     A list of control variables affecting the iterative BMA computations.
     A function called \code{iBMAcontrolLM} is provided to faciltate this
     setting, and the default is \code{iBMAcontrolLM(thresProbne0 = 5)}.
   }
  \item{verbose}{
     A logical variable indicating whether or not a detailed information 
     should be output as the computation progresses. The default value is
     \code{FALSE}.
  }
}
\value{
  A network represented as a data frame in which each row corresponds to a
  directed edge for which the probability is estimated to be nonzero. The 
  first column gives the name of the regulator, the second column gives the
  name of the regulated gene, and the third column gives the estimated
  probability for the regulator-gene pair. Rows are ordered by
  decreasing probability estimate. \cr
  The  \code{summary} function gives the number of inferred edges at 
  posterior probabilities 0, .5, .75, .90, .95 and 1.0
}
\details{
    \code{networkBMA} is intended for time-series data in which there are 
     more variables (gene expression values) than observations (experiments).
     For each gene, a linear model is fit to the expression data for all genes
     at a particular time point to predict the expression of a particular gene
     at the next time point. BMA is used to fit the linear model to identify
     the candidate regulators (variables) in the model. The inferred network 
     consists of candidate regulators and their corresponding posterior 
     probabilities for each gene. \cr
     It is assumed that data is available for all replicates at the same set
     of time points.
}
\section{References}{
  K. Lo, A. E. Raftery, K. M. Dombek, J. Zhu, E. E. Schadt, R. E. Bumgarner 
  and K. Y. Yeung (2012), Integrating External Biological 
  Knowledge in the Construction of Regulatory Networks from Time-series 
  Expression Data, BMC Systems Biology, 6:101.

  K. Y. Yeung, K. M. Dombek, K. Lo, J. E. Mittler, J. Zhu, E. E. Schadt,
  R. E. Bumgarner and A. E. Raftery (2011), Construction of 
  regulatory  networks  using expression time-series data of a genotyped 
  population, Proceedings of the National Academy of Sciences, 
  108(48):19436-41.

  K. Y. Yeung, A. E. Raftery and C. Fraley (2012),
  Uncovering regulatory relationships in yeast using \code{networkBMA},
  \code{networkBMA} Bioconductor package vignette.
}
\seealso{
\code{\link{summary.networkBMA}},
\code{\link{varord}},
\code{\link{iBMAcontrolLM}},
\code{\link{contabs}}
}
\examples{
data(dream4)

# there are a total of 5 datasets (networks) in the dream4ts10 data
network <- 1

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                           nTimePoints = nTimePoints, prior.prob = 0.01)

summary(edges1ts10)
}
\keyword{models}
