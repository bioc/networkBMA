\name{ScanBMA}
\alias{ScanBMA}
\title{
   Bayesian Model Averaging for linear regression models.
}
\description{
  Bayesian Model Averaging accounts for the model uncertainty inherent
  in the variable selection problem by averaging over the best models in
  the model class according to approximate posterior model
  probability.
}
\usage{
ScanBMA(x, y, prior.prob = NULL,
        control = ScanBMAcontrol(), verbose = FALSE) 
}
\arguments{
  \item{x}{
    A matrix of independent variables.
  }
  \item{y}{
    A vector of values for the dependent variable.
  }
  \item{prior.prob}{
    If included as input, either a single positive fraction representing the 
    probability of an independent variable being present in the true
    model, or else a vector assigning an estimated prior probability
    for each independent variable individually. The default value is
    \code{NULL}, which implies that no prior information will be used. 
  }
  \item{control}{
    A list of control variables affecting the ScanBMA computations.
    The function \code{ScanBMAcontrol} is provided to faciltate this
    setting, and the default is \code{ScanBMAcontrol()}. 
  }
  \item{verbose}{
     A logical variable indicating whether or not a detailed information 
     should be output as the computation progresses. The default value is
     \code{FALSE}.
  }
}
\value{
  Returns an object of class\code{bicreg} (see the \code{BMA}
  package). In addition, it adds \code{nmodelschecked}, which gives the
  number of models looked at in the ScanBMA model search, and \code{g},
  which gives the final value of g used if Zellner's g-prior was used to
  evaluate model likelihood.
}
\details{
  Bayesian Model Averaging accounts for the model uncertainty inherent
  in the variable selection problem by averaging over the best models in
  the model class according to approximate posterior model
  probability. ScanBMA is an algorithm for searching the model space
  efficiently when a large number of independent variables are present.
}
\section{References}{
  Raftery, Adrian E. (1995). Bayesian model selection in social research
  (with Discussion). Sociological Methodology 1995 (Peter V. Marsden,
  ed.), pp. 111-196, Cambridge, Mass.: Blackwells.

}
\seealso{
\code{\link{networkBMA}},
\code{\link{ScanBMAcontrol}},
\code{\link{gControl}}
}
\examples{
data(dream4)

# there are a total of 5 datasets (networks) in the dream4ts10 data
network <- 1

scanBMA.res <- ScanBMA( x = dream4ts10[[network]][,-(1:2)],
                        y = dream4ts10[[network]][,3],
                        prior.prob = 0.01)

summary(scanBMA.res)
}
\keyword{models}
