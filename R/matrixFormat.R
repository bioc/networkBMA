matrixFormat <-
function(x) 
{
 rownam <- rownames(x)
 array( as.vector(t(as.matrix(x))), c(2,2,nrow(x)), 
        dimnames = list(NULL,NULL, rownam))
}
