scores <-
function( contabs, what = c("TP", "FN", "FP", "TN", 
    "TPR", "TNR", "FPR", "FDR", "PPV", "NPV",
   "sensitivity", "specificity", "precision", "recall", 
    "F1", "MCC", "ACC", "expected", "O/E")) 
{

 TP <- contabs$TP
 TN <- contabs$TN
 FP <- contabs$FP
 FN <- contabs$FN

 names(TP) <- names(TN) <- names(FP) <- names(FN) <- rownames(contabs)

 TPR <- TP / (TP + FN) 

 sensitivity <- recall <- TPR

 FPR <- FP / (FP + TN)  

 ACC <- (TP+TN) / ((TP+FN)+(FP+TN))  

 specificity <- TNR <- 1- FPR

 precision <- PPV <- TP / (TP + FP)  
 precision[TP == 0 & FP == 0] <- 1

 NPV <- TN / (TN + FN)  

 FDR <- FP / (FP + TP)  
#FDR[TP == 0 & FP == 0] <- ?

 temp <- 2*(precision*recall)/(precision+recall)
 F1 <- 2*TP/(2*TP + FP + FN)

#print(max(abs(temp-F1score)))

# avoids overflow
 MCC <- (TP*TN - FP*FN) / (sqrt(TP+FN)*sqrt(FP+TN)*sqrt(TP+FP)*sqrt(FN+TN))

 expected <- ((TP + FP) * (TP + FN)) / (TP + FP + TN + FN)

 OE <- TP / expected

 similarity <- sqrt(PPV*specificity) #Lopes et al

 scr <- cbind.data.frame( TP = TP, FN = FN, FP = FP, TN = TN,
                   TPR = TPR, TNR = TNR, FPR = FPR, FDR = FDR,
                   sensitivity = sensitivity, specificity = specificity,
                   precision = precision, recall = recall,
                   PPV = PPV, NPV = NPV, F1 = F1, MCC = MCC, ACC = ACC, 
                   expected = expected, "O/E" = OE)[what]
 rownames(scr) <- rownames(contabs)
 scr
}
