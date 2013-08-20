\name{summary.networkBMA}
\alias{summary.networkBMA}
\title{
   Summarizes a \code{networkBMA} object.
}
\description{
   Counts the number of edges in a \code{networkBMA}  object that are
   at or above specified probability thresholds.
}
\usage{
\method{summary}{networkBMA}(object, thresholds = c(0,.5,.75,.90,.95,1), \dots)
}
\arguments{
  \item{object}{
    A \code{networkBMA} object.
  }
  \item{thresholds}{
    Threshold probabilities. The number of edges at or above these values
     are included in the output.
   }
 \item{\dots}{
    Not used. For generic/method consistency.
  }
}
\value{
  A vector giving the number of edges in the \code{object} that have
  probability at or above the values given in \code{thresholds}.
}
\seealso{
\code{\link{networkBMA}}
}
\examples{
data(dream4)

# there are a total of 5 datasets (networks) in the dream4ts100 data
network <- 1

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                           nTimePoints = nTimePoints, prior.prob = 0.01, 
                           nvar = 50)

summary(edges1ts10)
}
\keyword{models}
