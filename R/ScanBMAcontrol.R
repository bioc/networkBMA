ScanBMAcontrol <- function(OR = 100,
                           useg = TRUE,
                           gCtrl = gControl(),
                           thresProbne0 = 1) { 

  list( OR = OR, useg = useg, gCtrl = gCtrl,
        thresProbne0 = thresProbne0 );

}

gControl <- function(optimize = TRUE,
                     optMethod = "perTarget",
                     g0 = NULL,
                     iterlim = 100,
                     epsilon = 0.1 ) {
  
  list( optimize = optimize, g0 = g0,
        iterlim = iterlim, epsilon = epsilon );
}


