\name{writeEdges}
\alias{writeEdges}
\title{
  Output network edges to text in Cytoscape-readable format.
}
\description{
   Outputs a data frame of probabilistic network edges to a text file 
   suitable as input to Cytoscape for visualization.
}
\usage{
writeEdges( network, threshhold = .5, fileName = "edges.txt")
}
\arguments{
  \item{network}{
     A network represented as a data frame in which each row corresponds to 
     an edge for which the probability is estimated to be nonzero. The first
     column gives the name of the regulator, the second column gives the
     name of the regulated gene, and the third column gives the estimated
     probability for the regulator-gene pair.
  }
  \item{threshhold}{
    A threshhold on the probability that the variables in \code{x}.
    The default value is .5
   }
  \item{fileName}{
     A character string given the desired name of the output text file.
     The default is \code{"edges.txt"}.
  }
}
\value{
 The data frame of edges with those having probabilities below the specified
 threshhold removed. As a side effect, this data from is output to a text
 file named \code{fileName} in a format suitable for use with Cytoscape for
 visualization (see \url{www.cytoscape.org}).
}
\section{References}{
 Shannon P, Markiel A, Ozier O, Baliga NS, Wang JT, Ramage D, Amin N,
 Schwikowski B, Ideker T (2003), Cytoscape: a software environment for 
 integrated models of biomolecular interaction networks. Genome Research 
 13:2498-504
}
\seealso{
  \code{\link{networkBMA}}
}
\examples{
data(dream4)

network <- 1

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                          nTimePoints = nTimePoints, prior.prob = 0.1)

writeEdges(edges1ts10)
}
\keyword{models}
