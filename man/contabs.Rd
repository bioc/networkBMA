\name{contabs}
\alias{contabs}
\alias{matrixFormat}
\title{
   Contingency tables 
   for networks with probabilistic edges.
}
\description{
   Contingency table values relative to known truth 
   for probabilistic networks given thresholds
   on edge probabilies.
}
\usage{
contabs(network, reference, size, thresholds = NULL) 
}
\arguments{
  \item{network}{
     A network represented as a data frame in which each row corresponds to 
     an edge with nonzero estimated probability. The first
     column gives the name of the regulator, the second column gives the
     name of the regulated gene, and the third column gives the estimated
     probability for the regulator-gene pair. If the third column is omitted
     it is assumed that all edges are assigned probability 1.
  }
  \item{reference}{
     A reference network represented as a two-column data frame in which each 
     row corresponds to an edge, and the columns give the regulator and 
     target genes, respectively. This is the true underlying network
     for determining the contingency table values. 
  }
  \item{size}{
     The size of the network if all possible edges were included.
  }
  \item{thresholds}{
    Threshold values on the probability of edges being in the network for 
    which contingency tables are desired. The default is all distinct nonzero 
    probabilities in the network. 
    Thresholds should be specified as probabilities rather
    than percentages.
  }
}
\value{
   Outputs a data frame giving the contingency table values for the 
   specified thresholds, with row names being the threshold values.
   The variables (columns) are of the data frame  are as follows:  
   \item{TP}{
    true positives, or regulator-gene pairs 
     correctly identified as being linked in the reference network,
    } 
   \item{FN}{
    false negatives, or gene pairs that are linked in the reference network 
    but not in the proposed network,
    }
  \item{FP}{
    false positives, or gene pairs that are linked in the proposed network but 
    not in the reference network,
    }
  \item{TN}{
     true negatives, or gene pairs that are not linked both networks.
   } 
}
\details{
   Pairs for which edges do not exist in \code{reference} are not enumerated, 
   because their numbers could be very large. 
   Instead, the \code{size} parameter is used to
   account for pairs that are not edges. However, the assumption is that 
   all edges that are in the network but not in the reference are accounted 
   for in \code{size}. \cr
   Care must be taken in specifying \code{size}. For example, if the 
   reference does not contain self-edges while the network does, and  
  \code{size} does not count self-edges, then \code{contabs} 
   will not give the correct number of false negatives.
}
\seealso{
  \code{\link{networkBMA},\link{contabs.netwBMA}}
}
\examples{
data(dream4)

network <- 1

reference <- dream4gold10[[network]]
nGenes <- length(unique(c(reference[,1],reference[,2])))
nPossibleEdges <- nGenes^2

# no self loops in reference (there are none)
any(as.character(reference[,1]) == as.character(reference[,2]))

# restrict reference to edges; first two columns (gene) only
reference <- reference[reference[,3] == 1,1:2]

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10s <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                      nTimePoints = nTimePoints, prior.prob = 0.1, 
                      self = TRUE)

# self edges included in contingency table count
size <- nPossibleEdges

contingencyTables <- contabs(network = edges1ts10s, reference = reference,
                             size = size)

matrixFormat(contingencyTables)

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                      nTimePoints = nTimePoints, prior.prob = 0.1, 
                      self = FALSE)


# self edges excluded in contingency table count
size <- nPossibleEdges - nGenes

contingencyTables <- contabs(network = edges1ts10, reference = reference,
                             size = size)

matrixFormat(contingencyTables)
}
\keyword{models}
