"writeEdges" <-
function(network, threshhold = .5, fileName= "edges.txt") {
  if (any(network[,3] > 1)) stop("probability must not be given as percentage")
  use  <- network[,3] >= threshhold
  if (!any(use)) stop("no edges have probability at or above threshhold")
  network <- network[use,,drop= FALSE]
  write.table( network, file=fileName, quote=FALSE, 
               row.names=FALSE, col.names=FALSE, sep = "\t")
  invisible(network)
}
