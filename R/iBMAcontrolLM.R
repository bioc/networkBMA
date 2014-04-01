iBMAcontrolLM <- function( OR = 20, nbest = 10, 
                           maxNvar = 30, thresProbne0 = 1, keepModels = FALSE, 
                           maxIter = 200000) { 

  list( algorithm = "iBMA", OR = OR, nbest = nbest, maxNvar = maxNvar, 
        thresProbne0 = thresProbne0, keepModels = keepModels, maxIter = maxIter)

}



