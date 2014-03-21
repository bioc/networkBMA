\name{scores}
\alias{scores}
\title{
  Scores for assessment from contingency tables.
}
\description{
   Gives scores for assessment and other values associated with contingency
   tables for network inference.
}
\usage{
scores( contabs, what = c("TP", "FN", "FP", "TN", 
       "TPR", "TNR", "FPR", "FDR", "PPV", "NPV", 
       "sensitivity", "specificity", "precision", "recall", 
       "F1", "MCC", "ACC", "expected", "O/E")) 

}
\arguments{
  \item{contabs}{
    A data frame representing a contingency tables (confusion matrices) for
    a binary classification experiment. The column names should include
    \code{TP} (for true positive), \code{FN} (for false negative)
    \code{FP} (for false positive), \code{TN} (for true negative);
    these can be given in any order.
  }
  \item{what}{
      A character string specifying one or more desired output quantities
      from among:
     \describe{
        \item{"TP"}{
             Number of edges that are correctly identified as such 
            (true positives).
        }
        \item{"FP"}{
             Number of unlinked pairs that are incorrectly identified as edges
            (false positives).
        }
        \item{"TN"}{
             Number of unlinked pairs that are correctly identified as such
            (true negatives).
        }
        \item{"FN"}{
             Number of edges that are not identified as such
            (false negatives).
        }
        \item{"TPR","sensitivity","recall"}{
             True positive rate, sensitivity, or recall.
            \code{TP/(# edges)}.
        }
        \item{"FPR"}{
            False positive or false alarm rate.
            \code{FP/(# edges)}.
        }
        \item{"TNR","specificity"}{
             True negative rate or specificity.
            \code{TN/(# unlinked pairs)}.
        }
        \item{"FDR"}{
            False discovery rate.
            \code{FP/(# estimated links)}.
        }
        \item{"PPV","precision"}{
            Positive predictive value or precision.
            \code{TP/(# estimated links)}.
        }
        \item{"NPV"}{
            Negative predictive value.
            \code{TN/(# estimated unlinked pairs)}.
        }
        \item{"F1"}{
            F1 score, F-measure, balanced F-score.
            Harmonic mean of precision and recall.
        }
        \item{"MCC"}{
            Matthews correlation coefficient.
        }
        \item{"ACC"}{
            Accuracy.
            \code{(TP + TN)/(# edges + # unlinked pairs)}.
        }
        \item{"expected"}{
           Expected number of links under random assortment.
        }
        \item{"O/E"}{
            Ratio of estimated (observed) number of edges to
            expected number of links under random assortment.
            \code{TP/"expected"}.
        }
     }
  }
}
\value{
 A data frame in which the variables are the desired quantities derived
 the specified contingency tables. The rows correspond to the rows
 of the contingency tables supplied as input (\code{contabs}).
}
\seealso{
  \code{\link{contabs}},
  \code{\link{roc}},
  \code{\link{prc}}
}
\examples{
data(dream4)

network <- 1

reference <- dream4gold10[[network]]
nGenes <- length(unique(c(reference[,1],reference[,2])))
nPossibleEdges <- nGenes^2
reference <- reference[reference[,3] == 1,1:2]

nTimePoints <- length(unique(dream4ts10[[network]]$time))

edges1ts10 <- networkBMA( data = dream4ts10[[network]][,-(1:2)], 
                          nTimePoints = nTimePoints, prior.prob = 0.1,
                          self = FALSE)

size <- nPossibleEdges - nGenes

contingencyTables <- contabs(network = edges1ts10, reference = reference,
                             size = size)

scores(contingencyTables, what = c("sensitivity", "specificity", "FDR"))
}






















