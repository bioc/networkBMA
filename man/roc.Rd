 \name{roc}
\alias{roc}
\alias{prc}
\title{
   Receiver Operating Characteristic 
and Precision-Recall Curves
}
\description{
    Computes the    Receiver Operating Characteristic 
    (ROC) or precision-recall curves from contingencies tables
    (confusion matrices) and their associated probabilities, and optionally
    plots them.
}
\usage{
roc( contabs, plotit = TRUE)
prc( contabs, plotit = TRUE, ymax = max(y))
}
\arguments{
  \item{contabs}{
    A data frame representing a contingency tables (confusion matrices) for
    a binary classification experiment. The column names should include
    \code{TP} (for true positive), \code{FN} (for false negative),
    \code{FP} (for false positive), \code{TN} (for true negative);
    these can be given in any order.
  }
  \item{plotit}{
     A logical variable indicating whether or not the ROC curve should be
     plotted.
  }
  \item{ymax}{
     Upper vertical axis value for plotting precision-recall curves with 
     \code{prc}. The default is \code{max(y)}, (the maximum value for
     precision derived from the contingency table). \code{ymax} should be
     no larger than 1.
  }
}
\value{
   A vector with the following named components:
  \item{area}{
     The area under the extended curve (covered by the black solid and
      dotted lines).
  }
  \item{sector}{
    The estimated area under the sector (below the curve and within the red 
    lines) covered by the contingency table.
  }
  \item{width}{
    The width of the sector (within the red vertical lines)
    covered by the contingency table.
  }
  As a side-effect, the ROC (\code{roc}) or precision-recall curve 
  (\code{prc})  is plotted if \code{plotit = TRUE}. 
  The red vertical lines highlight the sector covered
  by the contingency table.
  The black lines approximate the curve. They are dotted
  beyond the endpoints of the sector, where there is no information
  from the contingency table. 
  The diagonal blue line in the ROC curve indicates the line between (0,0)
  and (1,1).
}
\details{
The estimated area may be inaccurate when the sector covered by the 
contingency tables is small, and should be used with caution in such cases. 
}

\section{References}{
J. Davis and M. Goadrich, The relationship between Precision-Recall and
ROC curves, manuscript, 
Department of Computer Science, University of Wisconsin.
}

\seealso{\code{\link{contabs}}}
\examples{
data(dream4)

network <- 1

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)],
                      nTimePoints = nTimePoints, prior.prob = 0.1)

# check for self loops in estimated network
selfN <- any(as.character(edges1ts10[,1]) == as.character(edges1ts10[,2]))
selfN

reference <- dream4gold10[[network]]

# check for self loops in reference (there are none)
selfR <- any(as.character(reference[,1]) == as.character(reference[,2]))
selfR

# restrict reference to edges; first two columns (gene) only
reference <- reference[reference[,3] == 1,1:2]

contingencyTables <- contabs(network = edges1ts10, reference = reference,
                             size = 100)

roc(contingencyTables)
prc(contingencyTables)
}


