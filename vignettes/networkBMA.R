### R code from vignette source 'networkBMA.Rnw'

###################################################
### code chunk number 1: vignette.data
###################################################
library(networkBMA)
data(vignette)

dim(timeSeries)
dim(reg.prob)
dim(brem.data)
reg.known

###################################################
### code chunk number 2: yeastScanBMA
###################################################
edges.ScanBMA <- networkBMA(data = timeSeries[,-(1:2)], 
                    nTimePoints = length(unique(timeSeries$time)), 
                    prior.prob = reg.prob,
                    nvar = 50,
                    ordering = "bic1+prior", diff100 = TRUE, diff0 = TRUE)
edges.ScanBMA[1:9,]


###################################################
### code chunk number 3: yeastiBMA
###################################################
edges.iBMA <- networkBMA(data = timeSeries[,-(1:2)], 
                          nTimePoints = length(unique(timeSeries$time)), 
                          prior.prob = reg.prob, known = reg.known,
                          nvar = 50, control = iBMAcontrolLM(),
                          ordering = "bic1+prior", diff100 = FALSE,
                          diff0 = FALSE)
edges.iBMA[1:9,]


###################################################
### code chunk number 4: contingency.tables
###################################################
ctables <- contabs.netwBMA( edges.ScanBMA, referencePairs, reg.known, 
                              thresh=c(.5,.75,.9))
ctables


###################################################
### code chunk number 5: scores
###################################################
scores( ctables, what = c("FDR", "precision", "recall"))


###################################################
### code chunk number 6: rocANDprc
###################################################
roc( contabs.netwBMA( edges.ScanBMA, referencePairs), plotit = TRUE)
title("ROC")

prc( contabs.netwBMA( edges.ScanBMA, referencePairs), plotit = TRUE)
title("Precision-Recall")


###################################################
### code chunk number 7: ScanBMA
###################################################
gene <- "YNL037C"
variables <- which(rownames(brem.data) != gene)

control <- ScanBMAcontrol(OR = 20, useg = TRUE,
                          gCtrl = gControl(optimize = FALSE, g0 = 20))

ScanBMAmodel.YNL037C <- ScanBMA( x = t(brem.data[variables,]),
                  y = unlist(brem.data[gene,]), control = control)


###################################################
### code chunk number 8: probne0
###################################################
ScanBMAmodel.YNL037C$probne0[ScanBMAmodel.YNL037C$probne0 > 0]


