\name{contabs.netwBMA}
\alias{contabs.netwBMA}
\title{
   Network assessment with incomplete context.
}
\description{
   Contingency table values for the intersection of a probabilistic network
   of regulator-target gene edges with a reference network with incomplete 
   knowledge.
}
\usage{
contabs.netwBMA(network, reference, known = NULL, threshholds = NULL)
}
\arguments{
  \item{network}{
     A network represented as a data frame in which each row corresponds to a
     directed edge (regulator-gene pair) with nonzero estimated probability. 
     The first column gives the name of the regulator, the second column gives
     the name of the regulated gene, and the third column gives the estimated
     probability for the regulator-gene pair. If the third column is omitted
     it is assumed that all edges are assigned probability 1.
  }
  \item{reference}{
     A reference network represented as a two-column data frame in which each 
     row corresponds to a regulator-gene pair (or network edge), and the 
     columns give the corresponding regulator and target gene names, 
     respectively. This reference network is used as the standard for 
     determining contingency table entries.
  }
 \item{known}{
    A 2-column matrix of regulatory relationships that were known (hard-coded)
    in the modeling process that produced \code{network}. Each row corresponds
    to a known regulator-gene pair (edge in the network), and the columns give
    the corresponding regulator and target gene names, respectively. 
    These known regulatory relationships should all be included in 
    \code{network} (this is checked), and they may also have regulator-gene 
    pairs in common with \code{reference}.
  }
  \item{threshholds}{
    Threshhold values on the probability of edges being in the network for 
    which contingency tables are desired. The default is all distinct nonzero 
    probabilities in the network. Nonzero threshholds are treated as 
    inclusive. Threshholds should be specified as probabilities rather
    than percentages.
  }
}
\value{
    The contingency table for the the intersection of \code{network} and 
   \code{reference} with edges in \code{known} removed.
}
\details{
     For real gene networks, complete information on the true underlying 
     network is not available. \code{contabs.netwBMA} proceeds by comparing
     the intersecting subset of the proposed edges with a reference network 
     composed of regulator-genes pairs.
   Edges that don't exist between any regulator and any target gene in the 
   reference network are used to complete the larger set P 
   of all possible edges for the comparison.
   Only the subset of network edges that belong to P is considered.
   There are issues in handling known edges hardcoded in the modeling,
   for which a detailed explanation is given in the package vignette.
}
\section{References}{
  Uncovering regulatory relationships in yeast using \code{networkBMA},
  \code{networkBMA} Bioconductor package vignette.
}

\seealso{
 \code{\link{networkBMA},\link{contabs.prelim},\link{contabs}}
}
\examples{
\dontrun{
data(vignette)

dim(timeSeries)
colnames(timeSeries)
table(timeSeries$replicate)
table(timeSeries$time)

dim(reg.known)
colnames(reg.known)

dim(reg.prob)

edges <- networkBMA(data = timeSeries[,-(1:2)],
                    nTimePoints = length(unique(timeSeries$time)),
                    prior.prob = reg.prob, known = reg.known)

contabs.netwBMA( network = edges, reference = referenceNetwork, 
                 known = reg.known)
}
}
\keyword{models}
