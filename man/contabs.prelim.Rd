\name{contabs.prelim}
\alias{contabs.prelim}
\title{
   Preliminary calculation for network assessment.
}
\description{
   Computes the quantities to be used for assessment from a proposed and
   a set of reference edges. For cases where there is incomplete information
   about the true underlying network.
}
\usage{
contabs.prelim(network, reference, known = NULL)
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
}
\value{
 A list with the following components:
 \item{network}{
     A network represented as a data frame in which each row corresponds to 
     an edge with nonzero estimated probability. This is the input 
     \code{network} with edges from the known regulatory relationships
     and edges outside of the domain defined by the output version of
    \code{reference} removed.
  }
  \item{reference}{
     A reference network represented as a two-column data frame in which each 
     row corresponds to an edge, and the columns give the regulator and 
     target genes, respectively. This is the input reference reduced by
     the known regulatory relationships.
  }
  \item{size}{
     The size of the network formed by pairs (r,g) in which r is any regulator
     g is any gene from the output version of \code{reference}, reduced by
     any known regulatory relationships.
  }
}
\details{
  This function accomplishes the preprocessing step from 
  \code{contabs.newtBMA} to produce the appropriate input to
  \code{contabs} to produce contingency tables.
}
\seealso{
 \code{\link{networkBMA},\link{contabs.netwBMA},\link{contabs}}
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

prelim <- contabs.prelim( network = edges, reference = referenceNetwork, 
                       known = reg.known)
contabs( network = prelim$network, reference = prelim$reference,
         size = prelim$size, threshholds = prelim$threshholds)

contabs.netwBMA( network = edges, reference = referenceNetwork, 
                 known = reg.known)
}
}
\keyword{models}
